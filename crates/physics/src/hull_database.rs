//! b3HullMap: hull byte identity, shared clones, and reference counts. Shape records hold the
//! retained b3HullData pointer; snapshot restore relocates these pointers in kernel records.

unsafe fn image(key: usize) -> &'static [u64] {
    let words = *((key as *const u32).add(35)) as usize / 8;
    core::slice::from_raw_parts(key as *const u64, words)
}
unsafe fn release(key: usize) {
    drop(Box::from_raw(core::ptr::slice_from_raw_parts_mut(
        key as *mut u64,
        image(key).len(),
    )));
}
const HOME: u16 = 0x0800;
const LINK: u16 = 0x07ff;
const FRAGMENT: u16 = 0xf000;
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Bucket {
    key: usize,
    refs: u32,
}
struct Database {
    table: Vec<u64>,
    buckets: &'static mut [Bucket],
    metadata: &'static mut [u16],
    count: usize,
    #[cfg(target_arch = "wasm32")]
    upload: Vec<u64>,
}
impl Database {
    const fn new() -> Self {
        Self {
            table: Vec::new(),
            buckets: &mut [],
            metadata: &mut [],
            count: 0,
            #[cfg(target_arch = "wasm32")]
            upload: Vec::new(),
        }
    }
    fn hash(&self, key: usize) -> u64 {
        unsafe { *((key as *const u64).add(1)) }
    }
    fn mask(&self) -> usize {
        self.buckets.len() - 1
    }
    fn next(&self, home: usize, bucket: usize) -> usize {
        let d = (self.metadata[bucket] & LINK) as usize;
        (home + d * (d + 1) / 2) & self.mask()
    }
    fn find(&self, bytes: &[u64]) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        let hash = bytes[1];
        let home = hash as usize & self.mask();
        if self.metadata[home] & HOME == 0 {
            return None;
        }
        let fragment = ((hash >> 48) as u16) & FRAGMENT;
        let mut bucket = home;
        loop {
            if self.metadata[bucket] & FRAGMENT == fragment
                && unsafe { image(self.buckets[bucket].key) } == bytes
            {
                return Some(bucket);
            }
            if self.metadata[bucket] & LINK == LINK {
                return None;
            }
            bucket = self.next(home, bucket);
        }
    }
    fn empty(&self, home: usize) -> Option<(usize, u16)> {
        for d in 1..LINK as usize {
            let bucket = (home + d * (d + 1) / 2) & self.mask();
            if self.metadata[bucket] == 0 {
                return Some((bucket, d as u16));
            }
        }
        None
    }
    fn predecessor(&self, home: usize, displacement: u16) -> usize {
        let mut bucket = home;
        while self.metadata[bucket] & LINK <= displacement {
            bucket = self.next(home, bucket);
        }
        bucket
    }
    fn evict(&mut self, bucket: usize) -> bool {
        let home = self.hash(self.buckets[bucket].key) as usize & self.mask();
        let mut prev = home;
        while self.next(home, prev) != bucket {
            prev = self.next(home, prev);
        }
        self.metadata[prev] = (self.metadata[prev] & !LINK) | (self.metadata[bucket] & LINK);
        let Some((empty, d)) = self.empty(home) else {
            return false;
        };
        prev = self.predecessor(home, d);
        self.buckets[empty] = self.buckets[bucket];
        self.metadata[empty] = (self.metadata[bucket] & FRAGMENT) | (self.metadata[prev] & LINK);
        self.metadata[prev] = (self.metadata[prev] & !LINK) | d;
        true
    }
    fn insert_raw(&mut self, value: Bucket) -> bool {
        if self.buckets.is_empty() || self.count + 1 > (self.buckets.len() as f64 * 0.9) as usize {
            return false;
        }
        let hash = self.hash(value.key);
        let fragment = ((hash >> 48) as u16) & FRAGMENT;
        let home = hash as usize & self.mask();
        if self.metadata[home] & HOME == 0 {
            if self.metadata[home] != 0 && !self.evict(home) {
                return false;
            }
            self.buckets[home] = value;
            self.metadata[home] = fragment | HOME | LINK;
        } else {
            let Some((empty, d)) = self.empty(home) else {
                return false;
            };
            let prev = self.predecessor(home, d);
            self.buckets[empty] = value;
            self.metadata[empty] = fragment | (self.metadata[prev] & LINK);
            self.metadata[prev] = (self.metadata[prev] & !LINK) | d;
        }
        self.count += 1;
        true
    }
    fn grow(&mut self) {
        let old_table = core::mem::take(&mut self.table);
        let old_buckets = core::mem::take(&mut self.buckets);
        let old_metadata = core::mem::take(&mut self.metadata);
        let mut size = (old_buckets.len() * 2).max(8);
        loop {
            self.allocate_table(size);
            self.count = 0;
            let mut success = true;
            for i in 0..old_buckets.len() {
                if old_metadata[i] != 0 && !self.insert_raw(old_buckets[i]) {
                    success = false;
                    break;
                }
            }
            if success {
                drop(old_table);
                return;
            }
            size *= 2;
        }
    }
    fn allocate_table(&mut self, size: usize) {
        let bucket_bytes = size * core::mem::size_of::<Bucket>();
        self.table = vec![0; (bucket_bytes + (size + 4) * 2).div_ceil(8)];
        unsafe {
            self.buckets =
                core::slice::from_raw_parts_mut(self.table.as_mut_ptr() as *mut Bucket, size);
            self.metadata = core::slice::from_raw_parts_mut(
                (self.table.as_mut_ptr() as *mut u8).add(bucket_bytes) as *mut u16,
                size + 4,
            );
            self.metadata[size] = 1;
        }
    }
    fn add(&mut self, bytes: &[u64]) -> usize {
        if let Some(bucket) = self.find(bytes) {
            self.buckets[bucket].refs += 1;
            return self.buckets[bucket].key;
        }
        let key = Box::into_raw(bytes.to_vec().into_boxed_slice()) as *mut u64 as usize;
        let bucket = Bucket { key, refs: 1 };
        while !self.insert_raw(bucket) {
            self.grow();
        }
        key
    }
    fn remove(&mut self, key: usize) {
        let bucket = self
            .find(unsafe { image(key) })
            .expect("hull database reference");
        self.buckets[bucket].refs -= 1;
        if self.buckets[bucket].refs != 0 {
            return;
        }
        let home = self.hash(key) as usize & self.mask();
        self.count -= 1;
        if self.metadata[bucket] & LINK == LINK {
            if bucket != home {
                let mut prev = home;
                while self.next(home, prev) != bucket {
                    prev = self.next(home, prev);
                }
                self.metadata[prev] |= LINK;
            }
            self.metadata[bucket] = 0;
        } else {
            let mut last = bucket;
            loop {
                let prev = last;
                last = self.next(home, last);
                if self.metadata[last] & LINK == LINK {
                    self.buckets[bucket] = self.buckets[last];
                    self.metadata[bucket] =
                        (self.metadata[bucket] & !FRAGMENT) | (self.metadata[last] & FRAGMENT);
                    self.metadata[prev] |= LINK;
                    self.metadata[last] = 0;
                    break;
                }
            }
        }
        unsafe {
            release(key);
        }
    }
}
impl Drop for Database {
    fn drop(&mut self) {
        for i in 0..self.buckets.len() {
            if self.metadata[i] != 0 {
                unsafe {
                    release(self.buckets[i].key);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_chains_share_by_bytes_and_survive_eviction_removal_and_rehash() {
        let mut db = Database::new();
        let mut images = Vec::new();
        let mut handles = Vec::new();
        for i in 0..300u64 {
            let mut image = vec![0u64; 18];
            image[1] = (i % 17) << 48;
            image[17] = (144 << 32) | i;
            let handle = db.add(&image);
            assert_eq!(db.add(&image), handle);
            images.push(image);
            handles.push(handle);
        }
        assert_eq!(db.count, 300);
        for i in (0..300).step_by(2) {
            db.remove(handles[i]);
            assert_eq!(db.buckets[db.find(&images[i]).unwrap()].refs, 1);
            db.remove(handles[i]);
            assert!(db.find(&images[i]).is_none());
        }
        for i in (1..300).step_by(2) {
            assert_eq!(db.add(&images[i]), handles[i]);
            for _ in 0..3 {
                db.remove(handles[i]);
            }
        }
        assert_eq!(db.count, 0);
        assert!(db.metadata[..db.buckets.len()].iter().all(|m| *m == 0));
    }
}

#[cfg(target_arch = "wasm32")]
pub use runtime::*;
#[cfg(target_arch = "wasm32")]
mod runtime {
    use super::*;
    use crate::regions::{self, MAX_WORLDS};
    static mut DATABASES: [Database; MAX_WORLDS] = [const { Database::new() }; MAX_WORLDS];
    #[export_name = "hullUploadBuffer"]
    pub extern "C" fn upload_buffer(world: usize, bytes: usize) -> *mut u64 {
        unsafe {
            let upload = &mut DATABASES[world].upload;
            upload.resize(bytes.div_ceil(8), 0);
            upload.as_mut_ptr()
        }
    }
    #[export_name = "hullDatabaseAdd"]
    pub extern "C" fn add(world: usize, bytes: usize) -> usize {
        unsafe {
            let db = &mut DATABASES[world];
            let input = core::slice::from_raw_parts(db.upload.as_ptr(), bytes / 8);
            db.add(input)
        }
    }
    #[export_name = "hullDatabaseLookup"]
    pub extern "C" fn lookup(world: usize, bytes: usize) -> i32 {
        unsafe {
            let db = &DATABASES[world];
            let input = core::slice::from_raw_parts(db.upload.as_ptr(), bytes / 8);
            db.find(input).map_or(-1, |i| db.buckets[i].key as i32)
        }
    }
    #[export_name = "hullDatabaseRemove"]
    pub extern "C" fn remove(world: usize, handle: usize) {
        unsafe {
            DATABASES[world].remove(handle);
        }
    }
    #[export_name = "hullDatabaseCount"]
    pub extern "C" fn count(world: usize) -> usize {
        unsafe { DATABASES[world].count }
    }
    #[export_name = "hullDatabaseRefs"]
    pub extern "C" fn refs(world: usize, handle: usize) -> u32 {
        unsafe {
            let db = &DATABASES[world];
            db.find(image(handle)).map_or(0, |i| db.buckets[i].refs)
        }
    }
    pub unsafe fn reset(world: usize) {
        DATABASES[world] = Database::new();
    }

    pub unsafe fn snapshot(world: usize, out: &mut Vec<u8>) {
        let db = &DATABASES[world];
        regions::write_word(out, db.buckets.len());
        for i in 0..db.buckets.len() {
            regions::write_word(out, db.metadata[i] as usize);
            if db.metadata[i] == 0 {
                continue;
            }
            regions::write_word(out, db.buckets[i].key);
            regions::write_word(out, db.buckets[i].refs as usize);
            let hull = image(db.buckets[i].key);
            regions::write_word(out, hull.len());
            out.extend_from_slice(core::slice::from_raw_parts(
                hull.as_ptr() as *const u8,
                hull.len() * 8,
            ));
        }
        regions::write_word(out, db.count);
    }
    pub unsafe fn restore(world: usize, input: &mut &[u8]) {
        reset(world);
        let db = &mut DATABASES[world];
        let n = regions::read_word(input);
        if n != 0 {
            db.allocate_table(n);
        }
        let mut relocations = Vec::new();
        for i in 0..n {
            db.metadata[i] = regions::read_word(input) as u16;
            if db.metadata[i] == 0 {
                continue;
            }
            let old = regions::read_word(input) as u32;
            db.buckets[i].refs = regions::read_word(input) as u32;
            let words = regions::read_word(input);
            let mut hull = vec![0u64; words];
            core::ptr::copy_nonoverlapping(input.as_ptr(), hull.as_mut_ptr() as *mut u8, words * 8);
            *input = &input[words * 8..];
            let pointer = Box::into_raw(hull.into_boxed_slice()) as *mut u64 as usize;
            db.buckets[i].key = pointer;
            relocations.push((old, pointer as u32));
        }
        db.count = regions::read_word(input);
        relocations.sort_unstable_by_key(|r| r.0);
        crate::shapes::relocate_hulls(world, &relocations);
    }
}
