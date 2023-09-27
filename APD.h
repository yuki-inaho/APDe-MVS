#ifndef _APD_H_
#define _APD_H_

#include "main.h"

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)
#define M_PI 3.14159265358979323846

using namespace boost::filesystem;

void CudaSafeCall(const cudaError_t error, const std::string &file, const int line);

void CudaCheckError(const char *file, const int line);

bool ReadBinMat(const path &mat_path, cv::Mat &mat);

bool WriteBinMat(const path &mat_path, const cv::Mat &mat, bool flush = false);

bool ReadCamera(const path &cam_path, Camera &cam);

bool ReadImage(const path &img_path, cv::Mat &img);

bool ShowDepthMap(const path &depth_path, const cv::Mat &depth, float depth_min, float depth_max);

bool ShowConfidenceMap(const path &confidence_path, const cv::Mat &confidence);

bool ShowNormalMap(const path &normal_path, const cv::Mat &normal);

bool ShowWeakImage(const path &weak_path, const cv::Mat &weak);

bool ExportPointCloud(const path &point_cloud_path, std::vector<PointList> &pointcloud, bool export_color = true);

std::string ToFormatIndex(int index);

template<typename TYPE>
void RescaleMatToTargetSize(const cv::Mat &src, cv::Mat &dst, const cv::Size2i &target_size);

void RunFusion(const path &dense_folder, const std::vector<Problem> &problems, const std::string &name = "APD.ply");

void RunFusion_TAT_A(const path &dense_folder, const std::vector<Problem> &problems,
                     const std::string &name = "APD.ply");

void RunFusion_TAT_I(const path &dense_folder, const std::vector<Problem> &problems,
                     const std::string &name = "APD.ply");

void DepthCompareToWeak(const path &dense_folder, const std::vector<Problem> &problems);

void CounterMap(const path &dense_folder, const std::vector<Problem> &problems);

void ShowCounterMap(const path &counter_path, const cv::Mat &counter_map);


struct cudaTextureObjects {
    cudaTextureObject_t images[MAX_IMAGES];
};

struct DataPassHelper {
    int width;
    int height;
    int ref_index;
    cudaTextureObjects *texture_objects_cuda;
    cudaTextureObjects *texture_depths_cuda;
    Camera *cameras_cuda;
    float4 *plane_hypotheses_cuda;
    curandState *rand_states_cuda;
    unsigned int *selected_views_cuda;
    short2 *anchors_cuda;
    int *anchors_map_cuda;
    uchar *weak_info_cuda;
    uchar *confidence_cuda;
    uchar *sa_mask_cuda;
    float *costs_cuda;
    PatchMatchParams *params;
    int2 debug_point;
    float4 *fit_plane_hypotheses_cuda;
    uchar *weak_reliable_cuda;
    uchar *view_weight_cuda;
    short2 *weak_nearest_strong;
};

class APD {
public:
    APD(const Problem &problem);

    ~APD();

    void InuputInitialization();

    void CudaSpaceInitialization();

    void SetDataPassHelperInCuda();

    void RunPatchMatch();

    float4 GetPlaneHypothesis(int r, int c);

    cv::Mat GetPixelStates();

    cv::Mat GetConfidence();

    int GetWidth();

    int GetHeight();

    float GetDepthMin();

    float GetDepthMax();

private:
    void ExportFitNormal();

    void ExportAnchors();

    void ExportNearestStrong();

    int num_images;
    int width;
    int height;
    Problem problem;
    // =========================
    // image host and cuda
    std::vector<cv::Mat> images;
    cudaTextureObjects texture_objects_host;
    cudaArray *cuArray[MAX_IMAGES];
    cudaTextureObjects *texture_objects_cuda;
    // =========================
    // depth host and cuda
    std::vector<cv::Mat> depths;
    cudaTextureObjects texture_depths_host;
    cudaArray *cuDepthArray[MAX_IMAGES];
    cudaTextureObjects *texture_depths_cuda;
    // =========================
    // camera host and cuda
    std::vector<Camera> cameras;
    Camera *cameras_cuda;
    // =========================
    // weak info host and cuda
    int weak_count;
    cv::Mat weak_info_host;
    uchar *weak_info_cuda;
    uchar *weak_reliable_cuda;
    short2 *weak_nearest_strong_cuda;
    cv::Mat confidence_host;
    uchar *confidence_cuda;
    // =========================
    // sa mask
    cv::Mat sa_mask_host;
    uchar *sa_mask_cuda;
    // =========================
    // anchor host and cuda
    short2 *anchors_cuda;
    cv::Mat anchors_map_host;
    int *anchors_map_cuda;
    // =========================
    // plane hypotheses host and cuda
    std::shared_ptr<float4[]> plane_hypotheses_host;
    float4 *plane_hypotheses_cuda;
    float4 *fit_plane_hypotheses_cuda;
    // =========================
    // cost cuda
    float *costs_cuda;
    // =========================
    // other var
    // params
    PatchMatchParams params_host;
    PatchMatchParams *params_cuda;
    // random states
    curandState *rand_states_cuda;
    // vis info
    // cv::Mat selected_views_host;
    unsigned int *selected_views_cuda;
    // for easy data pass
    DataPassHelper helper_host;
    DataPassHelper *helper_cuda;
    // save view weigth
    uchar *view_weight_cuda;
    //export for test
#ifdef DEBUG_COST_LINE
    float *weak_ncc_cost_cuda;
#endif // DEBUG_COST_LINE
};

class MemoryCache {
public:
    static std::shared_ptr<MemoryCache> get_instance();

    std::unordered_map<std::string, cv::Mat> img_cache;
    std::unordered_map<std::string, cv::Mat> mat_cache;
    std::unordered_map<std::string, Camera> cam_cache;
private:
    MemoryCache() = default;
};

#endif // !_APD_H_