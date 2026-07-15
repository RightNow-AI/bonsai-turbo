// cupti_shim — LD_PRELOAD CUDA kernel telemetry (device-side truth).
//
// Counts every kernel execution (including CUDA-graph-launched ones) and sums
// their durations via the CUPTI activity API, in-process — no profiler daemon,
// works in unprivileged containers where nsys stalls.
//
// Build: gcc -shared -fPIC tools/cupti_shim.c \
//          -I/usr/local/cuda/extras/CUPTI/include \
//          -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti -o cupti_shim.so
// Use:   CUPTI_SHIM_OUT=stats.json LD_PRELOAD=./cupti_shim.so ./app
//
// Output JSON: {"kernels": N, "kernel_ns": T, "memops": M, "memop_ns": U,
//               "span_ns": S}
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long g_kernels = 0, g_kernel_ns = 0;
static unsigned long long g_memops = 0, g_memop_ns = 0;
static unsigned long long g_first_ts = 0, g_last_ts = 0;

#define BUF_SIZE (8 * 1024 * 1024)

static void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size,
                                      size_t* max_num_records) {
    *buffer = (uint8_t*)malloc(BUF_SIZE + 8);
    *size = BUF_SIZE;
    *max_num_records = 0;
}

static void track_span(unsigned long long start, unsigned long long end) {
    if (g_first_ts == 0 || start < g_first_ts) g_first_ts = start;
    if (end > g_last_ts) g_last_ts = end;
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t stream_id,
                                      uint8_t* buffer, size_t size,
                                      size_t valid_size) {
    CUpti_Activity* record = NULL;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                CUpti_ActivityKernel4* k = (CUpti_ActivityKernel4*)record;
                g_kernels++;
                g_kernel_ns += k->end - k->start;
                track_span(k->start, k->end);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                CUpti_ActivityMemcpy* m = (CUpti_ActivityMemcpy*)record;
                g_memops++;
                g_memop_ns += m->end - m->start;
                track_span(m->start, m->end);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET: {
                CUpti_ActivityMemset* m = (CUpti_ActivityMemset*)record;
                g_memops++;
                g_memop_ns += m->end - m->start;
                track_span(m->start, m->end);
                break;
            }
            default:
                break;
        }
    }
    free(buffer);
}

__attribute__((constructor)) static void shim_init(void) {
    cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
}

__attribute__((destructor)) static void shim_fini(void) {
    cuptiActivityFlushAll(1);
    const char* path = getenv("CUPTI_SHIM_OUT");
    FILE* f = path ? fopen(path, "w") : stderr;
    if (!f) f = stderr;
    fprintf(f,
            "{\"kernels\": %llu, \"kernel_ns\": %llu, \"memops\": %llu, "
            "\"memop_ns\": %llu, \"span_ns\": %llu}\n",
            g_kernels, g_kernel_ns, g_memops, g_memop_ns,
            g_last_ts - g_first_ts);
    if (path && f != stderr) fclose(f);
}
