// Per-process and per-thread CPU timeline for same-user processes, without privileges.
// usage: cputrace <out.ndjson> <interval_us> <pattern>...   (runs until SIGTERM/SIGINT)
// CPUTRACE_EXCLUDE="pid,pid" leaves out processes that belong to other clients, such as another app's WebKit.
// A line per sample per process whose CPU advanced: {"t":epoch_us,"pid":..,"name":..,"cpu":ns_delta,"threads":[[name,ns],...]}
#include <libproc.h>
#include <mach/mach_time.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/proc_info.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#define MAXP 256
#define MAXT 512
static volatile int stop = 0;
static void onsig(int s) { (void)s; stop = 1; }
typedef struct { pid_t pid; char name[64]; unsigned long long cpu; int nt; unsigned long long tid[MAXT]; unsigned long long tcpu[MAXT]; } P;
static P ps[MAXP];
static int np = 0;
static mach_timebase_info_data_t tb;
static unsigned long long ns(unsigned long long abs) { return abs * tb.numer / tb.denom; }
static long long now_us(void) { struct timeval tv; gettimeofday(&tv, 0); return (long long)tv.tv_sec * 1000000 + tv.tv_usec; }

static pid_t excluded[MAXP];
static int nexcluded = 0;
static void refresh(char **pats, int npat) {
    pid_t pids[4096];
    int n = proc_listallpids(pids, sizeof pids) ;
    for (int i = 0; i < n; i++) {
        char path[PROC_PIDPATHINFO_MAXSIZE];
        if (proc_pidpath(pids[i], path, sizeof path) <= 0) continue;
        int hit = 0;
        for (int k = 0; k < npat; k++) if (strstr(path, pats[k])) hit = 1;
        if (!hit) continue;
        int known = 0;
        for (int j = 0; j < nexcluded; j++) if (excluded[j] == pids[i]) known = 1;
        for (int j = 0; j < np; j++) if (ps[j].pid == pids[i]) known = 1;
        if (known || np >= MAXP) continue;
        struct rusage_info_v4 r;
        if (proc_pid_rusage(pids[i], RUSAGE_INFO_V4, (rusage_info_t *)&r) != 0) continue;
        P *p = &ps[np++];
        memset(p, 0, sizeof *p);
        p->pid = pids[i];
        const char *base = strrchr(path, '/');
        snprintf(p->name, sizeof p->name, "%s", base ? base + 1 : path);
        p->cpu = ns(r.ri_user_time + r.ri_system_time);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: cputrace out interval_us pattern...\n"); return 2; }
    mach_timebase_info(&tb);
    signal(SIGTERM, onsig);
    signal(SIGINT, onsig);
    const char *exclude = getenv("CPUTRACE_EXCLUDE");
    for (const char *c = exclude; c && *c && nexcluded < MAXP;) {
        excluded[nexcluded++] = (pid_t)strtol(c, (char **)&c, 10);
        if (*c == ',') c++;
        else break;
    }
    FILE *out = fopen(argv[1], "w");
    int interval = atoi(argv[2]);
    char **pats = argv + 3;
    int npat = argc - 3;
    long long lastRefresh = 0;
    while (!stop) {
        long long t = now_us();
        if (t - lastRefresh > 20000) { refresh(pats, npat); lastRefresh = t; }
        for (int j = 0; j < np; j++) {
            P *p = &ps[j];
            if (p->pid == 0) continue;
            struct rusage_info_v4 r;
            if (proc_pid_rusage(p->pid, RUSAGE_INFO_V4, (rusage_info_t *)&r) != 0) { p->pid = 0; continue; }
            unsigned long long c = ns(r.ri_user_time + r.ri_system_time);
            if (c == p->cpu) continue;
            fprintf(out, "{\"t\":%lld,\"pid\":%d,\"name\":\"%s\",\"cpu\":%llu,\"threads\":[", t, p->pid, p->name, c - p->cpu);
            p->cpu = c;
            uint64_t tids[MAXT];
            int bytes = proc_pidinfo(p->pid, PROC_PIDLISTTHREADS, 0, tids, sizeof tids);
            int first = 1;
            for (int k = 0; k < bytes / (int)sizeof(uint64_t); k++) {
                struct proc_threadinfo ti;
                if (proc_pidinfo(p->pid, PROC_PIDTHREADINFO, tids[k], &ti, sizeof ti) <= 0) continue;
                unsigned long long tc = ti.pth_user_time + ti.pth_system_time; // already ns
                unsigned long long prev = 0; int slot = -1;
                for (int m = 0; m < p->nt; m++) if (p->tid[m] == tids[k]) { slot = m; prev = p->tcpu[m]; }
                if (slot < 0 && p->nt < MAXT) { slot = p->nt++; p->tid[slot] = tids[k]; prev = tc; /* first sight: no delta */ }
                if (slot >= 0) p->tcpu[slot] = tc;
                if (tc > prev) {
                    char nm[MAXTHREADNAMESIZE * 2];
                    int o = 0;
                    for (int q = 0; ti.pth_name[q] && o < (int)sizeof nm - 2; q++) { char ch = ti.pth_name[q]; if (ch == '"' || ch == '\\') nm[o++] = '\\'; nm[o++] = ch; }
                    nm[o] = 0;
                    fprintf(out, "%s[\"%s#%llu\",%llu]", first ? "" : ",", nm, (unsigned long long)tids[k], tc - prev);
                    first = 0;
                }
            }
            fprintf(out, "]}\n");
        }
        fflush(out);
        long long spent = now_us() - t;
        if (spent < interval) usleep(interval - spent);
    }
    fclose(out);
    return 0;
}
