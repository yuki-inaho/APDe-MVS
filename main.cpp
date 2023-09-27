#include "main.h"
#include "APD.h"

using namespace boost::filesystem;
namespace opt = boost::program_options;

opt::variables_map ParseArgs(int argc, char **argv) {
    opt::options_description desc("Allowed options");
    desc.add_options()
            ("dense_folder,d", opt::value<std::string>()->required(), "path to dense folder")
            ("gpu_index,g", opt::value<int>()->default_value(0), "gpu index")
            ("dataset,D", opt::value<std::string>()->default_value(std::string("DTU")),
             "dataset name, DTU, ETH3D or Tanks and Temples")
            ("only_fuse,f", opt::value<bool>()->default_value(false), "only fuse depths")
            ("no_fuse,F", opt::value<bool>()->default_value(false), "skip fuse")
            ("memory_cache,m", opt::value<bool>()->default_value(true), "use memory cache")
            ("flush", opt::value<bool>()->default_value(false), "Flush mat to disk")
            ("export_anchor,n", opt::value<bool>()->default_value(false), "Export anchor points to disk")
            ("help,h", "produce help message");

    opt::variables_map vm;
    try {
        opt::store(opt::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            exit(0);
        }
        opt::notify(vm);
    }
    catch (opt::error &e) {
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << desc << std::endl;
        exit(-1);
    }
    return vm;
}


void GenerateSampleList(const path &dense_folder, std::vector<Problem> &problems) {
    path cluster_list_path = dense_folder / path("pair.txt");
    path image_folder = dense_folder / path("images");
    problems.clear();
    ifstream file(cluster_list_path);
    std::stringstream iss;
    std::string line;
    std::vector<std::string> support_ext = {".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"};

    int num_images;
    iss.clear();
    std::getline(file, line);
    iss.str(line);
    iss >> num_images;

    for (int i = 0; i < num_images; ++i) {
        Problem problem;
        problem.src_image_ids.clear();
        iss.clear();
        std::getline(file, line);
        iss.str(line);
        iss >> problem.ref_image_id;

        problem.dense_folder = dense_folder;
        problem.result_folder = dense_folder / path("APD") / path(ToFormatIndex(problem.ref_image_id));
        create_directory(problem.result_folder);

        int num_src_images;
        iss.clear();
        std::getline(file, line);
        iss.str(line);
        iss >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            iss >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            problem.src_image_ids.push_back(id);
        }
        // get image's ext
        std::string ext;
        for (auto &support: support_ext) {
            path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + support);
            if (exists(image_path)) {
                ext = support;
                break;
            }
        }
        if (ext.empty()) {
            std::cout << "Error: can not find image: " << ToFormatIndex(problem.ref_image_id) << std::endl;
            exit(-1);
        }
        problem.img_ext = ext;
        problems.push_back(problem);
    }
}

bool CheckImages(const std::vector<Problem> &problems) {
    if (problems.size() == 0) {
        return false;
    }
    path image_path = problems[0].dense_folder / path("images") /
                      path(ToFormatIndex(problems[0].ref_image_id) + problems[0].img_ext);
    cv::Mat image;
    if (!ReadImage(image_path, image)) {
        return false;
    }
    const int width = image.cols;
    const int height = image.rows;
    for (size_t i = 1; i < problems.size(); ++i) {
        image_path = problems[i].dense_folder / path("images") /
                     path(ToFormatIndex(problems[i].ref_image_id) + problems[i].img_ext);
        if (!ReadImage(image_path, image)) {
            return false;
        }
        if (image.cols != width || image.rows != height) {
            return false;
        }
    }
    return true;
}

int ComputeRoundNum(const std::vector<Problem> &problems) {
    if (problems.size() == 0) {
        return 0;
    }
    path image_path = problems[0].dense_folder / path("images") /
                      path(ToFormatIndex(problems[0].ref_image_id) + problems[0].img_ext);
    cv::Mat image;
    if (!ReadImage(image_path, image)) {
        return 0;
    }
    int max_size = MAX(image.cols, image.rows);
    int round_num = 1;
    while (max_size > 800) {  // 800 for TAT & BlendedMVS & DTU and 1000 for ETH3D
        max_size /= 2;
        round_num++;
    }
    return round_num;
}

void ProcessProblem(const Problem &problem) {
    std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..."
              << std::endl;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    APD APD(problem);
    APD.InuputInitialization();
    APD.CudaSpaceInitialization();
    APD.SetDataPassHelperInCuda();
    APD.RunPatchMatch();

    int width = APD.GetWidth(), height = APD.GetHeight();
    cv::Mat depth = cv::Mat(height, width, CV_32FC1);
    cv::Mat normal = cv::Mat(height, width, CV_32FC3);
    cv::Mat pixel_states = APD.GetPixelStates();
    cv::Mat confidence = APD.GetConfidence();
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float4 plane_hypothesis = APD.GetPlaneHypothesis(r, c);
            depth.at<float>(r, c) = plane_hypothesis.w;
            if (depth.at<float>(r, c) < APD.GetDepthMin() || depth.at<float>(r, c) > APD.GetDepthMax()) {
                depth.at<float>(r, c) = 0;
                pixel_states.at<uchar>(r, c) = UNKNOWN;
            }
            normal.at<cv::Vec3f>(r, c) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
        }
    }

    path depth_path = problem.result_folder / path("depths.bin");
    WriteBinMat(depth_path, depth);
    path normal_path = problem.result_folder / path("normals.bin");
    WriteBinMat(normal_path, normal);
    path weak_path = problem.result_folder / path("weak.bin");
    WriteBinMat(weak_path, pixel_states);

    if (problem.params.geom_consistency || problem.params.use_APD) {
        path confidence_path = problem.result_folder / path("confidence.bin");
        WriteBinMat(confidence_path, confidence);
    }

    if (problem.show_medium_result) {
        path depth_img_path = problem.result_folder / path("depth_" + std::to_string(problem.iteration) + ".jpg");
        path normal_img_path = problem.result_folder / path("normal_" + std::to_string(problem.iteration) + ".jpg");
        path weak_img_path = problem.result_folder / path("weak_" + std::to_string(problem.iteration) + ".png");

        ShowDepthMap(depth_img_path, depth, APD.GetDepthMin(), APD.GetDepthMax());
        ShowNormalMap(normal_img_path, normal);
        ShowWeakImage(weak_img_path, pixel_states);
        if (problem.params.geom_consistency || problem.params.use_APD) {
            path confidence_img_path = problem.result_folder / path("confidence_" + std::to_string(problem.iteration) +
                                                                    ".png");
            ShowConfidenceMap(confidence_img_path, confidence);
        }

    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!"
              << std::endl;
    std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms"
              << std::endl;
}

int main(int argc, char **argv) {
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Parse arguments and prepare for processing
    ////////////////////////////////////////////////////////////////////////////////////////////////
    opt::variables_map vm = ParseArgs(argc, argv);
    std::string dense_folder_str = vm["dense_folder"].as<std::string>();
    int gpu_index = vm["gpu_index"].as<int>();
    std::string dataset = vm["dataset"].as<std::string>();
    bool only_fuse = vm["only_fuse"].as<bool>();
    bool no_fuse = vm["no_fuse"].as<bool>();
    bool use_memory_cache = vm["memory_cache"].as<bool>();
    bool flush = vm["flush"].as<bool>();
    bool export_anchor = vm["export_anchor"].as<bool>();
    // it is not necessary to use memory cache when only_fuse is true
    if (only_fuse) {
        use_memory_cache = false;
    }
    // it is necessary to flush memory cache to disk when no_fuse is true
    if (no_fuse) {
        flush = true;
    }
    // show config
    std::cout << "========================== Config ==========================" << std::endl;
    std::cout << "dense_folder : " << dense_folder_str << std::endl;
    std::cout << "gpu_index    : " << gpu_index << std::endl;
    std::cout << "dataset      : " << dataset << std::endl;
    std::cout << "only_fuse    : " << only_fuse << std::endl;
    std::cout << "no_fuse      : " << no_fuse << std::endl;
    std::cout << "memory_cache : " << use_memory_cache << std::endl;
    std::cout << "flush        : " << flush << std::endl;
    std::cout << "export_anchor: " << export_anchor << std::endl;
    std::cout << "============================================================" << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // set environment for processing
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // use memory cache can speed up the program, but it will load all data into the memory
    // including all images, depths, normals, weaks, confidences, selected_views, cameras.
    if (use_memory_cache) {
        printf("Use memory cache!\n");
        auto memory_cache = MemoryCache::get_instance();
    }
    path dense_folder(dense_folder_str);
    path output_folder = dense_folder / path("APD");
    create_directory(output_folder);
    cudaSetDevice(gpu_index);
    // generate problems
    std::vector<Problem> problems;
    GenerateSampleList(dense_folder, problems);
    if (!CheckImages(problems)) {
        std::cout << "Images may error, check it!\n";
        return EXIT_FAILURE;
    }
    std::cout << "There are " << problems.size() << " problems needed to be processed!" << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // if only_fuse is true, then only do fusion
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (only_fuse) {
        if (dataset == "TaT_a") {
            RunFusion_TAT_A(dense_folder, problems, "APD.ply");
        } else if (dataset == "TaT_i") {
            RunFusion_TAT_I(dense_folder, problems, "APD.ply");
        } else {
            RunFusion(dense_folder, problems, "APD.ply");
        }
        printf("Fusion done!\n");
        return EXIT_SUCCESS;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // compute round num and set params
    ////////////////////////////////////////////////////////////////////////////////////////////////
    int round_num = ComputeRoundNum(problems);
    std::cout << "Round nums: " << round_num << std::endl;
    // init common problem params
    for (auto &problem: problems) {
        if (dataset == "TaT_a" || dataset == "TaT_i") {
            problem.params.geom_factor = 0.05f;
        } else {
            problem.params.geom_factor = 0.2f;
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // iteration for each round
    ////////////////////////////////////////////////////////////////////////////////////////////////
    int iteration_index = 0;
    const int geom_iteration = 3;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < round_num; ++i) {
        std::cout << "========================== Round " << i << " ==========================" << std::endl;
        std::cout << "======== iteration " << iteration_index << "========" << std::endl;
        for (auto &problem: problems) {
            {
                auto &params = problem.params;
                if (i == 0) {
                    params.state = FIRST_INIT;
                    params.use_APD = false;
                } else {
                    params.state = REFINE_INIT;
                    params.use_APD = true;
                    params.ransac_threshold = 0.01 - i * 0.00125;
                    params.rotate_time = MIN(static_cast<int>(std::pow(2, i)), 4);
                }
                params.geom_consistency = false;
                params.max_iterations = 3;
                params.weak_peak_radius = 6;
            }
            problem.show_medium_result = false;
            problem.iteration = iteration_index;
            problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale
            ProcessProblem(problem);
        }
        iteration_index++;
        std::cout << "======== iteration " << iteration_index << "========" << std::endl;
        for (int j = 0; j < geom_iteration; ++j) {
            bool is_last_iteration = (i == round_num - 1 && j == geom_iteration - 1);
            for (auto &problem: problems) {
                {
                    auto &params = problem.params;
                    params.state = REFINE_ITER;
                    if (i == 0) {
                        params.use_APD = false;
                    } else {
                        params.use_APD = true;
                        params.ransac_threshold = 0.01 - i * 0.00125;
                        params.rotate_time = MIN(static_cast<int>(std::pow(2, i)), 4);
                    }
                    params.geom_consistency = true;
                    params.max_iterations = 3;
                    params.weak_peak_radius = MAX(4 - 2 * j, 2);
                }
                if (is_last_iteration && export_anchor) {
                    problem.export_anchor = true;
                }
                problem.show_medium_result = (j == geom_iteration - 1);
                problem.iteration = iteration_index;
                problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale
                ProcessProblem(problem);
            }
            iteration_index++;
        }
        std::cout << "=============================================================" << std::endl;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms"
              << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // flush memory cache to disk
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (use_memory_cache && flush) {
        printf("Write memory cache to disk!\n");
        auto memory_cache = MemoryCache::get_instance();
        for (auto &mat_info: memory_cache->mat_cache) {
            auto &mat_path = mat_info.first;
            auto &mat = mat_info.second;
            WriteBinMat(mat_path, mat, true);
        }
        printf("All done!\n");
        memory_cache->mat_cache.clear();
        memory_cache->img_cache.clear();
        memory_cache->cam_cache.clear();
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // fusion or not
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (no_fuse) {
        printf("Skip fusion, all done!\n");
        return EXIT_SUCCESS;
    }
    std::cout << "Run fusion\n";
    if (dataset == "TaT_a") {
        RunFusion_TAT_A(dense_folder, problems, "APD.ply");
    } else if (dataset == "TaT_i") {
        RunFusion_TAT_I(dense_folder, problems, "APD.ply");
    } else {
        RunFusion(dense_folder, problems, "APD.ply");
    }
    std::cout << "All done\n";
    return EXIT_SUCCESS;
}
