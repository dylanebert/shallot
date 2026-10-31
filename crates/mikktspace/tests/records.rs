use std::{fs, path::Path};

#[test]
fn every_record_equals_c_bits_through_the_shipped_boundary() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/core/mesh/fixtures");
    let mut count = 0;
    for entry in fs::read_dir(&fixtures).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap();
        if !name.ends_with(".input.bin") {
            continue;
        }
        let bytes = fs::read(&path).unwrap();
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| u32::from_le_bytes(b))
            .collect();
        let (nv, nc) = (words[0] as usize, words[1] as usize);
        let mut input = vec![nv as u32, nc as u32, 1, 0, 8];
        input.extend_from_slice(&words[2..]);
        let reference = fs::read(fixtures.join(name.replace(".input.bin", ".bin"))).unwrap();
        // SAFETY: staging dimensions and indices come from the records generator; output
        // length is the first word of this module's allocation, which is freed once.
        unsafe {
            let ptr = shallot_mikktspace::generate(input.as_ptr(), input.len());
            assert!(!ptr.is_null(), "{name}");
            let out = std::slice::from_raw_parts(ptr, *ptr as usize);
            let n = out[1] as usize;
            let indices = &out[2 + n * 8..2 + n * 8 + nc];
            let tangents = &out[2 + n * 8 + nc..];
            let corners: Vec<u8> = indices
                .iter()
                .flat_map(|&i| {
                    tangents[i as usize * 4..i as usize * 4 + 4]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                })
                .collect();
            assert_eq!(corners, reference, "{name}");
            shallot_mikktspace::free_words(ptr, out.len());
        }
        count += 1;
    }
    assert!(count >= 15);
}
