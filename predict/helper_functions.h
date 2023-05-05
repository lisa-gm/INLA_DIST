#ifndef helperfunctions_h
#define helperfunction_h

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sched.h>
#include <string.h>
#include <errno.h>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime_api.h"

//#define DEBUG

//Mit diesen 3 Werte kannst du dir die NUMA Node ID holen:
inline int topo_get_numNode(int GPU_rank)
{

    //printf("in topo get numNode\n");
    int numaNodeID;
    int pciBus, pciDomain, pciDevice;
    cudaDeviceGetAttribute(&pciBus, cudaDevAttrPciBusId, GPU_rank);
    cudaDeviceGetAttribute(&pciDevice, cudaDevAttrPciDeviceId, GPU_rank);
    cudaDeviceGetAttribute(&pciDomain, cudaDevAttrPciDomainId, GPU_rank);

    //printf("pciBus: %d, pciDevice %d, pciDomain: %d\n", pciBus, pciDevice, pciDomain);

    char fname[1024];
    char buff[100];
    int ret = snprintf(fname, 1023, "/sys/bus/pci/devices/0000:%02x:%02x.%1x/numa_node", pciBus, pciDevice, pciDomain);
    if (ret > 0)
    {
        fname[ret] = '\0';
        FILE* fp = fopen(fname, "r");
        if (fp)
        {
            ret = fread(buff, sizeof(char), 99, fp);
            int numaNodeID = atoi(buff);
            fclose(fp);
            //std::cout << "In get NUMA ID. GPU rank: " << GPU_rank << ", pciBus: " << pciBus << ", pciDevice: " << pciDevice << ", pciDomain: " << pciDomain << ", numa node ID: " << numaNodeID << std::endl;

            return numaNodeID;
        }
    }
    return -1;
}

// given a numa domain -> identify all corresponding hwthreads/cores
// returns # of threads and writes corresponding indices into hwthread list
inline int read_numa_threads(int numa_node, int** hwthread_list)
{
    char path[1024];
    int total_hwthreads = sysconf(_SC_NPROCESSORS_CONF);
    int cpuidx = 0;
    int* cpulist = new int[total_hwthreads];
    if (!cpulist)
    {
        return -1;
    }

    for (int i = 0; i < total_hwthreads; i++)
    {
        int ret = snprintf(path, 1023, "/sys/devices/system/node/node%d/cpu%d", numa_node, i);
        if (!access(path, F_OK))
        {
#ifdef DEBUG
            printf("HWthread %d located in NUMA domain %d\n", i, numa_node);
#endif
            cpulist[cpuidx++] = i;
        }
    }
    *hwthread_list = cpulist;
#ifdef DEBUG
    printf("NUMA domain %d has %d HWThreads\n", numa_node, cpuidx);
#endif
    return cpuidx;
}

// pin count-many threads from hwthread list
inline int pin_hwthreads(int count, int* hwthread_list)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
#ifdef DEBUG
    printf("Pinning to");
#endif
    for (int i = 0; i < count; i++)
    {
        CPU_SET(hwthread_list[i], &cpuset);
#ifdef DEBUG
        printf(" %d", hwthread_list[i]);
#endif
    }
#ifdef DEBUG
    printf("\n");
#endif
    return sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}


#endif