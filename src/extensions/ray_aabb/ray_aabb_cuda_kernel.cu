#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

__global__ void ray_aabb_cuda_forward_kernel(
        const float* __restrict__ ray_dir,
        const float* __restrict__ voxel_bound,
        const int* __restrict__ ray_bid,
        const int* __restrict__ voxel_bid,
        int* __restrict__ mask,
        float* __restrict__ dist,
        const int ray_num,
        const int ray_feat_len,
        const int voxel_feat_len) {
    // voxel index
    const int voxel_idx = blockIdx.y;
    // ray index
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx < ray_num){
        // ray and voxel not belong to the same example, skip.
        if (ray_bid[ray_idx] != voxel_bid[voxel_idx])
            return;
        // store last in and first out.
        float tmin_max, tmax_min;
        // compute txmin, txmax
        float cur_bound_xmin, cur_bound_xmax, txmin, txmax;
        float cur_ray_invdir_x = 1 / (ray_dir[ray_idx * ray_feat_len + 0]+1e-12);
        if (cur_ray_invdir_x >= 0){
            cur_bound_xmin = voxel_bound[voxel_idx*voxel_feat_len+0];
            cur_bound_xmax = voxel_bound[voxel_idx*voxel_feat_len+3];
        }
        else{
            cur_bound_xmin = voxel_bound[voxel_idx*voxel_feat_len+3];
            cur_bound_xmax = voxel_bound[voxel_idx*voxel_feat_len+0];
        }
        txmin = cur_bound_xmin * cur_ray_invdir_x;
        txmax = cur_bound_xmax * cur_ray_invdir_x;
        tmin_max = txmin;
        tmax_min = txmax;
        
        // compute tymin, tymax
        float cur_bound_ymin, cur_bound_ymax, tymin, tymax;
        float cur_ray_invdir_y = 1 / (ray_dir[ray_idx * ray_feat_len + 1]+1e-12);
        if (cur_ray_invdir_y >= 0){
            cur_bound_ymin = voxel_bound[voxel_idx*voxel_feat_len+1];
            cur_bound_ymax = voxel_bound[voxel_idx*voxel_feat_len+4];
        }
        else{
            cur_bound_ymin = voxel_bound[voxel_idx*voxel_feat_len+4];
            cur_bound_ymax = voxel_bound[voxel_idx*voxel_feat_len+1];
        }
        tymin = cur_bound_ymin * cur_ray_invdir_y;
        tymax = cur_bound_ymax * cur_ray_invdir_y;
        // x leave before y or y leave before x
        if((tmin_max > tymax) || (tmax_min < tymin))
            return;
        tmin_max = fmaxf(tmin_max, tymin);
        tmax_min = fminf(tmax_min, tymax);

        // compute tzmin, tzmax
        float cur_bound_zmin, cur_bound_zmax, tzmin, tzmax;
        float cur_ray_invdir_z = 1 / (ray_dir[ray_idx * ray_feat_len + 2]+1e-12);
        if (cur_ray_invdir_z >= 0){
            cur_bound_zmin = voxel_bound[voxel_idx*voxel_feat_len+2];
            cur_bound_zmax = voxel_bound[voxel_idx*voxel_feat_len+5];
        }
        else{
            cur_bound_zmin = voxel_bound[voxel_idx*voxel_feat_len+5];
            cur_bound_zmax = voxel_bound[voxel_idx*voxel_feat_len+2];
        }
        tzmin = cur_bound_zmin * cur_ray_invdir_z;
        tzmax = cur_bound_zmax * cur_ray_invdir_z;
        // z leave before xy or xy leave before z
        if((tmin_max > tzmax) || (tmax_min < tzmin))
            return;
        tmin_max = fmaxf(tmin_max, tzmin);
        tmax_min = fminf(tmax_min, tzmax);
        
        // at this step, we know they intersect. so update mask and dist
        mask[voxel_idx*ray_num+ray_idx] = 1;
        dist[voxel_idx*ray_num*2+ray_idx*2+0] = tmin_max;
        dist[voxel_idx*ray_num*2+ray_idx*2+1] = tmax_min;
    } // ray_idx < ray_num
}

} // namespace

std::vector<torch::Tensor> ray_aabb_cuda_forward(
        torch::Tensor ray_dir,
        torch::Tensor voxel_bound,
        torch::Tensor ray_bid,
        torch::Tensor voxel_bid) {
    
    const auto device = ray_dir.device();
    const int ray_num = ray_dir.size(0);
    const int ray_feat_len = ray_dir.size(1);
    const int voxel_num = voxel_bound.size(0);
    const int voxel_feat_len = voxel_bound.size(1);

    torch::Tensor mask = torch::zeros({voxel_num, ray_num}, torch::dtype(torch::kInt32).device(device));
    torch::Tensor dist = torch::zeros({voxel_num, ray_num, 2}, torch::dtype(torch::kFloat32).device(device));


    const int threads = 1024;
    const dim3 blocks((ray_num + threads - 1) / threads, voxel_num);


    ray_aabb_cuda_forward_kernel<<<blocks, threads>>>(
            ray_dir.data_ptr<float>(),
            voxel_bound.data_ptr<float>(),
            ray_bid.data_ptr<int>(),
            voxel_bid.data_ptr<int>(),
            mask.data_ptr<int>(),
            dist.data_ptr<float>(),
            ray_num,
            ray_feat_len,
            voxel_feat_len
    );

    return {mask, dist};
}
