#ifndef helperfunctions_h
#define helperfunction_h

#include "cuda.h"
#include "cuda_runtime_api.h"

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


#endif
