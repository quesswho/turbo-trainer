//! CPU backend. Compiles the generated CUDA C kernels as C++ against a small
//! compatibility shim, then runs the grid on the host with OpenMP.

use std::{
    alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error},
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    io::Write,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::runtime::{
    Dialect, Dim3,
    bindings::{DeviceProps, GemmConfig, GpuBindings},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Cpu;

pub type CpuError = String;

type CpuResult = Result<(), CpuError>;

/// Allocations are aligned so that the generated kernels may read them as
/// `float*`, and so that a vectorising compiler has something to work with.
const ALIGN: usize = 64;

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
pub struct CpuPtr {
    /// Must stay first: a launch argument is a pointer to this struct, and the
    /// kernel reads it as a plain pointer.
    ptr: *mut u8,
    bytes: usize,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct CpuKernel {
    func: *mut c_void,
    launch: *mut c_void,
}

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;

type LaunchFn = unsafe extern "C" fn(*mut c_void, c_uint, c_uint, c_uint, c_uint, *mut *mut c_void, c_uint);

fn layout(bytes: usize) -> Layout {
    Layout::from_size_align(bytes.max(1), ALIGN).expect("bad allocation layout")
}

fn last_dl_error() -> String {
    let err = unsafe { dlerror() };
    if err.is_null() {
        "unknown dynamic loader error".into()
    } else {
        unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned()
    }
}

/// Enough of the CUDA execution model to run the generated kernels on the host.
const SHIM: &str = r##"
#include <math.h>
#include <string.h>
#include <stdint.h>

struct uint3 { unsigned x, y, z; };
struct dim3 { unsigned x, y, z; };

#define BULLET_TLS __attribute__((tls_model("global-dynamic")))

static thread_local BULLET_TLS uint3 threadIdx = {0, 0, 0};
static thread_local BULLET_TLS uint3 blockIdx = {0, 0, 0};
static thread_local BULLET_TLS dim3 blockDim = {1, 1, 1};
static thread_local BULLET_TLS dim3 gridDim = {1, 1, 1};

#define __global__
#define __device__ static inline
#define __forceinline__

struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct int2 { int x, y; };
struct int4 { int x, y, z, w; };

static inline float2 make_float2(float x, float y) { float2 v; v.x = x; v.y = y; return v; }
static inline float4 make_float4(float x, float y, float z, float w) { float4 v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }
static inline int2 make_int2(int x, int y) { int2 v; v.x = x; v.y = y; return v; }
static inline int4 make_int4(int x, int y, int z, int w) { int4 v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }

static inline float max(float a, float b) { return a > b ? a : b; }
static inline float min(float a, float b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }
static inline int min(int a, int b) { return a < b ? a : b; }
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
static inline float __fdividef(float a, float b) { return a / b; }

static inline float atomicAdd(float* p, float v) {
    float old;
    #pragma omp atomic capture
    { old = *p; *p += v; }
    return old;
}

static inline int atomicAdd(int* p, int v) {
    int old;
    #pragma omp atomic capture
    { old = *p; *p += v; }
    return old;
}

typedef void (*K0)();
typedef void (*K1)(void*);
typedef void (*K2)(void*, void*);
typedef void (*K3)(void*, void*, void*);
typedef void (*K4)(void*, void*, void*, void*);
typedef void (*K5)(void*, void*, void*, void*, void*);
typedef void (*K6)(void*, void*, void*, void*, void*, void*);
typedef void (*K7)(void*, void*, void*, void*, void*, void*, void*);
typedef void (*K8)(void*, void*, void*, void*, void*, void*, void*, void*);
typedef void (*K9)(void*, void*, void*, void*, void*, void*, void*, void*, void*);
typedef void (*K10)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
typedef void (*K11)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
typedef void (*K12)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);

static inline void bullet_cpu_call(void* f, void** a, unsigned n) {
    switch (n) {
        case 0: ((K0)f)(); break;
        case 1: ((K1)f)(a[0]); break;
        case 2: ((K2)f)(a[0], a[1]); break;
        case 3: ((K3)f)(a[0], a[1], a[2]); break;
        case 4: ((K4)f)(a[0], a[1], a[2], a[3]); break;
        case 5: ((K5)f)(a[0], a[1], a[2], a[3], a[4]); break;
        case 6: ((K6)f)(a[0], a[1], a[2], a[3], a[4], a[5]); break;
        case 7: ((K7)f)(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); break;
        case 8: ((K8)f)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); break;
        case 9: ((K9)f)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); break;
        case 10: ((K10)f)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]); break;
        case 11: ((K11)f)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]); break;
        case 12: ((K12)f)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]); break;
        default: __builtin_trap();
    }
}

extern "C" void bullet_cpu_launch(void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bdx,
                                  void** args, unsigned nargs) {
    const long long blocks = (long long)gx * (long long)gy * (long long)gz;

    #pragma omp parallel
    {
        gridDim.x = gx; gridDim.y = gy; gridDim.z = gz;
        blockDim.x = bdx; blockDim.y = 1; blockDim.z = 1;

        #pragma omp for schedule(static)
        for (long long b = 0; b < blocks; b++) {
            blockIdx.x = (unsigned)(b % (long long)gx);
            blockIdx.y = (unsigned)((b / (long long)gx) % (long long)gy);
            blockIdx.z = (unsigned)(b / ((long long)gx * (long long)gy));

            for (unsigned t = 0; t < bdx; t++) {
                threadIdx.x = t;
                bullet_cpu_call(f, args, nargs);
            }
        }
    }
}
"##;

impl GpuBindings for Cpu {
    type Err = CpuError;
    type Dev = ();
    type Ptr = CpuPtr;
    type Ctx = ();
    type Stream = ();
    type BlasHandle = ();
    type Kernel = CpuKernel;
    type Module = *mut c_void;

    unsafe fn driver_init() -> CpuResult {
        Ok(())
    }

    unsafe fn device_get(_ordinal: c_int) -> CpuResult {
        Ok(())
    }

    unsafe fn device_props(_device: ()) -> Result<DeviceProps, CpuError> {
        Ok(DeviceProps {
            name: "CPU".into(),
            warp_size: None,
            stream_mem_alloc: false,
            vec_atomics: false,
            arch: None,
            dialect: Dialect::CudaHip,
        })
    }

    unsafe fn context_create(_device: ()) -> CpuResult {
        Ok(())
    }

    unsafe fn context_destroy(_device: ()) -> CpuResult {
        Ok(())
    }

    unsafe fn context_set(_ctx: ()) -> CpuResult {
        Ok(())
    }

    unsafe fn context_sync() -> CpuResult {
        Ok(())
    }

    unsafe fn context_malloc(bytes: usize) -> Result<CpuPtr, CpuError> {
        let layout = layout(bytes);
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Ok(CpuPtr { ptr, bytes })
    }

    unsafe fn context_free(dev_ptr: CpuPtr) -> CpuResult {
        unsafe { dealloc(dev_ptr.ptr, layout(dev_ptr.bytes)) };
        Ok(())
    }

    unsafe fn context_memset(dev_ptr: CpuPtr, bytes: usize, value: u8) -> CpuResult {
        assert!(bytes <= dev_ptr.bytes);
        unsafe { std::ptr::write_bytes(dev_ptr.ptr, value, bytes) };
        Ok(())
    }

    unsafe fn context_memcpy_d2h(dst: *mut c_void, src: CpuPtr, bytes: usize) -> CpuResult {
        assert!(bytes <= src.bytes);
        unsafe { std::ptr::copy_nonoverlapping(src.ptr, dst.cast::<u8>(), bytes) };
        Ok(())
    }

    unsafe fn context_memcpy_h2d(dst: CpuPtr, src: *const c_void, bytes: usize) -> CpuResult {
        assert!(bytes <= dst.bytes);
        unsafe { std::ptr::copy_nonoverlapping(src.cast::<u8>(), dst.ptr, bytes) };
        Ok(())
    }

    unsafe fn stream_create() -> CpuResult {
        Ok(())
    }

    unsafe fn stream_destroy(_stream: ()) -> CpuResult {
        Ok(())
    }

    unsafe fn stream_sync(_stream: ()) -> CpuResult {
        Ok(())
    }

    unsafe fn stream_malloc(_stream: (), bytes: usize) -> Result<CpuPtr, CpuError> {
        unsafe { Self::context_malloc(bytes) }
    }

    unsafe fn stream_free(_stream: (), dev_ptr: CpuPtr) -> CpuResult {
        unsafe { Self::context_free(dev_ptr) }
    }

    unsafe fn stream_memset(_stream: (), dev_ptr: CpuPtr, bytes: usize, value: u8) -> CpuResult {
        unsafe { Self::context_memset(dev_ptr, bytes, value) }
    }

    unsafe fn stream_memcpy_d2h(_stream: (), dst: *mut c_void, src: CpuPtr, bytes: usize) -> CpuResult {
        unsafe { Self::context_memcpy_d2h(dst, src, bytes) }
    }

    unsafe fn stream_memcpy_h2d(_stream: (), dst: CpuPtr, src: *const c_void, bytes: usize) -> CpuResult {
        unsafe { Self::context_memcpy_h2d(dst, src, bytes) }
    }

    unsafe fn kernel_load(_kernel: CpuKernel) -> CpuResult {
        Ok(())
    }

    unsafe fn kernel_destroy(_kernel: CpuKernel) -> CpuResult {
        Ok(())
    }

    unsafe fn kernel_launch(
        func: CpuKernel,
        _stream: (),
        gdim: Dim3,
        bdim: Dim3,
        args: &mut [*mut c_void],
        smem: c_uint,
    ) -> CpuResult {
        if smem != 0 {
            return Err("the CPU backend does not implement shared memory".into());
        }

        if func.func.is_null() || func.launch.is_null() {
            return Err("launching an unresolved kernel".into());
        }

        // Each entry points at a `CpuPtr`, whose first field is the allocation.
        let mut values: Vec<*mut c_void> =
            args.iter().map(|&arg| unsafe { *arg.cast::<*mut c_void>() }).collect();

        let launch = unsafe { std::mem::transmute::<*mut c_void, LaunchFn>(func.launch) };

        unsafe {
            launch(func.func, gdim.x, gdim.y, gdim.z, bdim.x, values.as_mut_ptr(), values.len() as c_uint)
        };

        Ok(())
    }

    unsafe fn module_create(code: *const c_void) -> Result<*mut c_void, CpuError> {
        let path = unsafe { CStr::from_ptr(code.cast::<c_char>()) };
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };

        if handle.is_null() {
            return Err(format!("could not load {}: {}", path.to_string_lossy(), last_dl_error()));
        }

        Ok(handle)
    }

    unsafe fn module_destroy(_module: *mut c_void) -> CpuResult {
        // Deliberately not `dlclose`d. The OpenMP pool inside a module outlives
        // the launch, and unloading it out from under those threads faults.
        Ok(())
    }

    unsafe fn module_get_kernel(module: *mut c_void, kernel_name: &CStr) -> Result<CpuKernel, CpuError> {
        let func = unsafe { dlsym(module, kernel_name.as_ptr()) };
        if func.is_null() {
            return Err(format!("no kernel named {}", kernel_name.to_string_lossy()));
        }

        let name = c"bullet_cpu_launch";
        let launch = unsafe { dlsym(module, name.as_ptr()) };
        if launch.is_null() {
            return Err("the compiled module has no launcher".into());
        }

        Ok(CpuKernel { func, launch })
    }

    unsafe fn program_compile(
        source_code: &CStr,
        _num_options: c_int,
        _options: *const *const c_char,
    ) -> Result<Vec<c_char>, CpuError> {
        static COUNT: AtomicUsize = AtomicUsize::new(0);

        let unit = COUNT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("bullet-cpu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

        let source_path = dir.join(format!("kernel{unit}.cpp"));
        let object_path = dir.join(format!("kernel{unit}.so"));

        let mut file = std::fs::File::create(&source_path).map_err(|e| e.to_string())?;
        file.write_all(SHIM.as_bytes()).map_err(|e| e.to_string())?;
        file.write_all(source_code.to_bytes()).map_err(|e| e.to_string())?;
        drop(file);

        let compiler = std::env::var("BULLET_CPU_CXX").unwrap_or_else(|_| "c++".into());

        let output = Command::new(&compiler)
            .args(["-O2", "-fPIC", "-shared", "-fopenmp", "-march=native", "-w"])
            .arg(&source_path)
            .arg("-o")
            .arg(&object_path)
            .output()
            .map_err(|e| format!("could not run {compiler}: {e}"))?;

        if !output.status.success() {
            return Err(format!(
                "{compiler} failed on {}:\n{}",
                source_path.display(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let path = CString::new(object_path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;

        Ok(path.as_bytes_with_nul().iter().map(|&b| b as c_char).collect())
    }

    unsafe fn blas_create() -> Result<(), CpuError> {
        Ok(())
    }

    unsafe fn blas_destroy(_handle: ()) -> CpuResult {
        Ok(())
    }

    unsafe fn blas_set_stream(_handle: (), _stream: ()) -> CpuResult {
        Ok(())
    }

    unsafe fn blas_gemm(_handle: (), config: GemmConfig, a: CpuPtr, b: CpuPtr, c: CpuPtr) -> CpuResult {
        unsafe { gemm(config, a, b, c, 1, 0) };
        Ok(())
    }

    unsafe fn blas_gemm_batched(
        _handle: (),
        batch_size: c_int,
        config: GemmConfig,
        a: CpuPtr,
        b: CpuPtr,
        c: CpuPtr,
    ) -> CpuResult {
        unsafe { gemm(config, a, b, c, batch_size.max(0) as usize, 1) };
        Ok(())
    }
}

/// Column major, matching the `cublasSgemm` call the CUDA backend makes:
/// `C = alpha * op(A) * op(B) + beta * C`, where a row major operand is the
/// transposed one.
unsafe fn gemm(config: GemmConfig, a: CpuPtr, b: CpuPtr, c: CpuPtr, batches: usize, strided: usize) {
    let (m, n, k) = (config.m.max(0) as usize, config.n.max(0) as usize, config.k.max(0) as usize);
    let (alpha, beta) = (config.alpha, config.beta);

    let stride_a = if strided == 1 { m * k } else { 0 };
    let stride_b = if strided == 1 { k * n } else { 0 };
    let stride_c = if strided == 1 { m * n } else { 0 };

    for batch in 0..batches.max(1) {
        let a = unsafe { std::slice::from_raw_parts(a.ptr.cast::<f32>().add(batch * stride_a), m * k) };
        let b = unsafe { std::slice::from_raw_parts(b.ptr.cast::<f32>().add(batch * stride_b), k * n) };
        let c = unsafe { std::slice::from_raw_parts_mut(c.ptr.cast::<f32>().add(batch * stride_c), m * n) };

        for j in 0..n {
            for i in 0..m {
                let mut sum = 0.0;
                for l in 0..k {
                    let av = if config.row_mjr_a { a[i * k + l] } else { a[l * m + i] };
                    let bv = if config.row_mjr_b { b[l * n + j] } else { b[j * k + l] };
                    sum += av * bv;
                }

                let target = &mut c[j * m + i];
                *target = alpha * sum + beta * *target;
            }
        }
    }
}
