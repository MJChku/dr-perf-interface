// Embed an rpath so the binary finds libperfmark.so without LD_LIBRARY_PATH.
fn main() {
    let dir = std::env::var("PERFMARK_DIR")
        .unwrap_or_else(|_| format!("{}/../../build", env!("CARGO_MANIFEST_DIR")));
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir);
    println!("cargo:rerun-if-env-changed=PERFMARK_DIR");
}
