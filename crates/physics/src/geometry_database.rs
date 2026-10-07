//! Per-world immutable mesh, height-field and compound uploads keyed by caller identity.
use crate::regions::{self, MAX_WORLDS};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, DefaultHasher};
type Index<K, V> = HashMap<K, V, BuildHasherDefault<DefaultHasher>>;

struct Entry {
    kind: u32,
    identity: u32,
    refs: u32,
    bytes: usize,
}
struct Database {
    entries: Index<usize, Entry>,
    identities: Index<(u32, u32), usize>,
    upload: Vec<u64>,
}
impl Database {
    const fn new() -> Self {
        Self {
            entries: HashMap::with_hasher(BuildHasherDefault::new()),
            identities: HashMap::with_hasher(BuildHasherDefault::new()),
            upload: Vec::new(),
        }
    }
    fn lookup(&self, kind: u32, identity: u32) -> usize {
        self.identities.get(&(kind, identity)).copied().unwrap_or(0)
    }
    fn entry(&self, kind: u32, pointer: usize) -> Option<&Entry> {
        self.entries.get(&pointer).filter(|e| e.kind == kind)
    }
    fn retain(&mut self, kind: u32, identity: u32, bytes: usize, refs: u32, payload: Vec<u64>) -> usize {
        let pointer = Box::into_raw(payload.into_boxed_slice()) as *mut u64 as usize;
        self.entries.insert(pointer, Entry { kind, identity, refs, bytes });
        self.identities.insert((kind, identity), pointer);
        pointer
    }
    fn add(&mut self, kind: u32, identity: u32, bytes: usize, refs: u32) -> usize {
        let pointer = self.lookup(kind, identity);
        if pointer != 0 {
            if bytes != 0 {
                self.upload.fill(0);
                self.upload.clear();
            }
            self.entries.get_mut(&pointer).unwrap().refs += refs;
            return pointer;
        }
        let words = bytes.div_ceil(8);
        assert!(bytes != 0 && self.upload.len() == words);
        let payload = core::mem::take(&mut self.upload);
        self.retain(kind, identity, bytes, refs, payload)
    }
    fn remove(&mut self, kind: u32, pointer: usize) {
        let entry = self.entries.get_mut(&pointer).expect("retained geometry");
        assert_eq!(entry.kind, kind);
        entry.refs -= 1;
        if entry.refs != 0 { return; }
        let entry = self.entries.remove(&pointer).unwrap();
        self.identities.remove(&(entry.kind, entry.identity));
        unsafe { release(pointer, entry.bytes); }
    }
    fn clear(&mut self) {
        self.upload.fill(0);
        drop(core::mem::take(&mut self.upload));
        for (pointer, entry) in core::mem::take(&mut self.entries) {
            unsafe { release(pointer, entry.bytes); }
        }
        self.identities = Index::default();
    }
    fn allocation_bytes(&self) -> usize {
        self.entries.capacity() * core::mem::size_of::<(usize, Entry)>()
            + self.identities.capacity() * core::mem::size_of::<((u32, u32), usize)>()
            + self.upload.capacity() * core::mem::size_of::<u64>()
            + self.entries.values().map(|entry| entry.bytes.div_ceil(8) * 8).sum::<usize>()
    }
}
unsafe fn release(pointer: usize, bytes: usize) {
    drop(Box::from_raw(core::ptr::slice_from_raw_parts_mut(pointer as *mut u64, bytes.div_ceil(8))));
}
static mut DATABASES: [Database; MAX_WORLDS] = [const { Database::new() }; MAX_WORLDS];

#[export_name = "geometryUploadBuffer"]
pub extern "C" fn upload_buffer(world: usize, bytes: usize) -> *mut u64 {
    unsafe {
        let db = &mut DATABASES[world];
        db.upload.resize(bytes.div_ceil(8), 0);
        db.upload.as_mut_ptr()
    }
}
#[export_name = "geometryDatabaseAdd"]
pub extern "C" fn add(world: usize, kind: u32, identity: u32, bytes: usize, refs: u32) -> usize {
    unsafe { DATABASES[world].add(kind, identity, bytes, refs) }
}
#[export_name = "geometryDatabaseLookup"]
pub extern "C" fn lookup(world: usize, kind: u32, identity: u32) -> usize {
    unsafe { DATABASES[world].lookup(kind, identity) }
}
#[export_name = "geometryDatabaseRemove"]
pub extern "C" fn remove(world: usize, kind: u32, pointer: usize) {
    unsafe { DATABASES[world].remove(kind, pointer); }
}
#[export_name = "geometryDatabaseIdentity"]
pub extern "C" fn identity(world: usize, kind: u32, pointer: usize) -> u32 {
    unsafe { DATABASES[world].entry(kind, pointer).map_or(0, |e| e.identity) }
}
#[export_name = "geometryDatabaseCount"]
pub extern "C" fn count(world: usize) -> usize {
    unsafe { DATABASES[world].entries.len() }
}
#[export_name = "geometryDatabaseAllocationBytes"]
pub extern "C" fn allocation_bytes(world: usize) -> usize {
    unsafe { DATABASES[world].allocation_bytes() }
}
#[export_name = "geometryDatabaseRefs"]
pub extern "C" fn refs(world: usize, kind: u32, pointer: usize) -> u32 {
    unsafe { DATABASES[world].entry(kind, pointer).map_or(0, |e| e.refs) }
}
pub unsafe fn reset(world: usize) {
    DATABASES[world].clear();
}
pub unsafe fn snapshot(world: usize, out: &mut Vec<u8>) {
    let db = &DATABASES[world];
    regions::write_word(out, db.entries.len());
    for (&pointer, e) in &db.entries {
        regions::write_word(out, e.kind as usize);
        regions::write_word(out, e.identity as usize);
        regions::write_word(out, e.refs as usize);
        regions::write_word(out, pointer);
        regions::write_word(out, e.bytes);
        out.extend_from_slice(core::slice::from_raw_parts(pointer as *const u8, e.bytes));
    }
}
pub unsafe fn restore(world: usize, input: &mut &[u8]) {
    reset(world);
    let db = &mut DATABASES[world];
    let count = regions::read_word(input);
    let mut relocations = Vec::with_capacity(count);
    for _ in 0..count {
        let kind = regions::read_word(input) as u32;
        let identity = regions::read_word(input) as u32;
        let refs = regions::read_word(input) as u32;
        let old = regions::read_word(input) as u32;
        let bytes = regions::read_word(input);
        let mut payload = vec![0u64; bytes.div_ceil(8)];
        core::ptr::copy_nonoverlapping(input.as_ptr(), payload.as_mut_ptr() as *mut u8, bytes);
        *input = &input[bytes..];
        let pointer = db.retain(kind, identity, bytes, refs, payload);
        relocations.push((old, pointer as u32));
    }
    relocations.sort_unstable_by_key(|pair| pair.0);
    crate::shapes::relocate_geometry(world, &relocations);
}
