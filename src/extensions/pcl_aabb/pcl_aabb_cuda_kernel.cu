#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

__global__ void pcl_aabb_cuda_forward_kernel(
        const float* __restrict__ pcl_pos,
        const float* __restrict__ voxel_bound,
        const int* __restrict__ pcl_bid,
        const int* __restrict__ voxel_bid,
        int* __restrict__ mask,
        const int pcl_num,
        const int pcl_feat_len,
        const int voxel_feat_len) {
    // voxel index
    const int voxel_idx = blockIdx.y;
    // pcl index
    const int pcl_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pcl_idx < pcl_num){
        // pcl and voxel not belong to the same example, skip.
        if (pcl_bid[pcl_idx] != voxel_bid[voxel_idx])
            return;
        float cur_pcl_pos_x = pcl_pos[pcl_idx * pcl_feat_len + 0];
        float cur_bound_xmin = voxel_bound[voxel_idx*voxel_feat_len+0];
        float cur_bound_xmax = voxel_bound[voxel_idx*voxel_feat_len+3];
        if ((cur_pcl_pos_x < cur_bound_xmin) || (cur_pcl_pos_x > cur_bound_xmax))
            return;
        float cur_pcl_pos_y = pcl_pos[pcl_idx * pcl_feat_len + 1];
        float cur_bound_ymin = voxel_bound[voxel_idx*voxel_feat_len+1];
        float cur_bound_ymax = voxel_bound[voxel_idx*voxel_feat_len+4];
        if ((cur_pcl_pos_y < cur_bound_ymin) || (cur_pcl_pos_y > cur_bound_ymax))
            return;
        float cur_pcl_pos_z = pcl_pos[pcl_idx * pcl_feat_len + 2];
        float cur_bound_zmin = voxel_bound[voxel_idx*voxel_feat_len+2];
        float cur_bound_zmax = voxel_bound[voxel_idx*voxel_feat_len+5];
        if ((cur_pcl_pos_z < cur_bound_zmin) || (cur_pcl_pos_z > cur_bound_zmax))
            return;
        // at this step, we know point inside voxel. set mask to 1.
        mask[voxel_idx*pcl_num+pcl_idx] = 1;
    } // pcl_idx < pcl_num
}

} // namespace

torch::Tensor pcl_aabb_cuda_forward(
        torch::Tensor pcl_pos,
        torch::Tensor voxel_bound,
        torch::Tensor pcl_bid,
        torch::Tensor voxel_bid) {
    
    const auto device = pcl_pos.device();
    const int pcl_num = pcl_pos.size(0);
    const int pcl_feat_len = pcl_pos.size(1);
    const int voxel_num = voxel_bound.size(0);
    const int voxel_feat_len = voxel_bound.size(1);

    torch::Tensor mask = torch::zeros({voxel_num, pcl_num}, torch::dtype(torch::kInt32).device(device));


    const int threads = 1024;
    const dim3 blocks((pcl_num + threads - 1) / threads, voxel_num);


    pcl_aabb_cuda_forward_kernel<<<blocks, threads>>>(
            pcl_pos.data_ptr<float>(),
            voxel_bound.data_ptr<float>(),
            pcl_bid.data_ptr<int>(),
            voxel_bid.data_ptr<int>(),
            mask.data_ptr<int>(),
            pcl_num,
            pcl_feat_len,
            voxel_feat_len
    );

    return mask;
}
