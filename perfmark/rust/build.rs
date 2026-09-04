// Link against libperfmark.so from the drperf build directory (or PERFMARK_DIR).
fn main() {
    let dir = std::env::var("PERFMARK_DIR")
        .unwrap_or_else(|_| format!("{}/../../build", env!("CARGO_MANIFEST_DIR")));
    println!("cargo:rustc-link-search=native={}", dir);
    println!("cargo:rerun-if-env-changed=PERFMARK_DIR");
    // Consumers read this to set an rpath for their binaries.
    println!("cargo:dir={}", dir);
}
