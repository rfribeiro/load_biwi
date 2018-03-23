#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <sstream>
namespace fs = std::experimental::filesystem;
using namespace std;
using namespace cv;

#define CROP_SIZE 192
const int debug = 0;

bool loadDepthImageCompressed(Mat& depthImg, const char* fname) {

	//now read the depth image
	FILE* pFile = fopen(fname, "rb");
	if (!pFile) {
		cerr << "could not open file " << fname << endl;
		return false;
	}

	int im_width = 0;
	int im_height = 0;
	bool success = true;

	success &= (fread(&im_width, sizeof(int), 1, pFile) == 1); // read width of depthmap
	success &= (fread(&im_height, sizeof(int), 1, pFile) == 1); // read height of depthmap

	depthImg.create(im_height, im_width, CV_16SC1);
	depthImg.setTo(0);


	int numempty;
	int numfull;
	int p = 0;

	if (!depthImg.isContinuous())
	{
		cerr << "Image has the wrong size! (should be 640x480)" << endl;
		return false;
	}

	int16_t* data = depthImg.ptr<int16_t>(0);
	while (p < im_width*im_height) {

		success &= (fread(&numempty, sizeof(int), 1, pFile) == 1);

		for (int i = 0; i < numempty; i++)
			data[p + i] = 0;

		success &= (fread(&numfull, sizeof(int), 1, pFile) == 1);
		success &= (fread(&data[p + numempty], sizeof(int16_t), numfull, pFile) == (unsigned int)numfull);
		p += numempty + numfull;

	}

	fclose(pFile);

	return success;
}

int16_t* loadDepthImageCompressed(const char* fname) {

	//now read the depth image
	FILE* pFile =fopen(fname, "rb");
	if (!pFile) {
		std::cerr << "could not open file " << fname << std::endl;
		return NULL;
	}

	int im_width = 0;
	int im_height = 0;
	bool success = true;

	success &= (fread(&im_width, sizeof(int), 1, pFile) == 1); // read width of depthmap
	success &= (fread(&im_height, sizeof(int), 1, pFile) == 1); // read height of depthmap

	int16_t* depth_img = new int16_t[im_width*im_height];

	int numempty;
	int numfull;
	int p = 0;

	while (p < im_width*im_height) {

		success &= (fread(&numempty, sizeof(int), 1, pFile) == 1);

		for (int i = 0; i < numempty; i++)
			depth_img[p + i] = 0;

		success &= (fread(&numfull, sizeof(int), 1, pFile) == 1);
		success &= (fread(&depth_img[p + numempty], sizeof(int16_t), numfull, pFile) == (unsigned int)numfull);
		p += numempty + numfull;

	}

	fclose(pFile);

	if (success)
		return depth_img;
	else {
		delete[] depth_img;
		return NULL;
	}
}

bool load_and_save_depth_image(string fname, int x, int y) {

	//int16_t* img = loadDepthImageCompressed(fname.c_str());

	//cv::Mat A(480, 640, CV_16S, img);
	cv::Mat depthImg;
	//read depth image (compressed!)
	if (!loadDepthImageCompressed(depthImg, fname.c_str()))
		return false;

	string file_pattern = fname.substr(0, fname.length() - 4);

	// crop image
	cv::Rect crop = cv::Rect(x - CROP_SIZE / 2+10, y - CROP_SIZE / 2, CROP_SIZE , CROP_SIZE );
	cv::Mat cropImg = depthImg(crop);

	cv::Mat depth8u = cropImg.clone();
	depth8u.convertTo(depth8u, CV_8UC1, 255.0 / 1000);

	equalizeHist(depth8u, depth8u);

	cv::Mat falseColorsMap;
	applyColorMap(depth8u, falseColorsMap, cv::COLORMAP_RAINBOW);

	cv::Mat jetColorsMap;
	applyColorMap(depth8u, jetColorsMap, cv::COLORMAP_JET);

	cv::Mat hsvColorsMap;
	applyColorMap(depth8u, hsvColorsMap, cv::COLORMAP_HSV);

	if (debug) {
		imshow("teste", depth8u);
		imshow("jet", jetColorsMap);
		imshow("aut", falseColorsMap);
		imshow("hsv", hsvColorsMap);

		cvWaitKey(0);
	}
	
	//imwrite((file_pattern + "_crop.png").c_str(),depth8u);
	imwrite((file_pattern + "_face.png").c_str(), jetColorsMap);
	imwrite((file_pattern + "_face2.png").c_str(), falseColorsMap);

	return true;
}

void load_and_save_rgb_image(string fname, int x, int y) 
{

	string file_pattern = fname.substr(0, fname.length() - 4);

	cv::Mat A = cv::imread(fname.c_str());

	// crop image
	cv::Rect crop = cv::Rect(x- CROP_SIZE/2, y- CROP_SIZE/2, CROP_SIZE, CROP_SIZE);
	cv::Mat cropImg = A(crop);

	if (debug) {
		imshow("teste", cropImg);
		cvWaitKey(0);
	}
	imwrite((file_pattern + "_face.png").c_str(),cropImg);
}

bool read_gt(float* gt, const char* fname) {

	//try to read in the ground truth from a binary file
	FILE* pFile = fopen(fname, "rb");
	if (!pFile) {
		std::cerr << "could not open file " << fname << std::endl;
		return NULL;
	}

	float* data = new float[6];

	bool success = true;
	success &= (fread(&data[0], sizeof(float), 6, pFile) == 6);
	fclose(pFile);

	if (success) {
		memcpy(gt, data, 6 * sizeof(float));
	}

	delete[] data;
	return success;
}

bool read_calibration(float* cal, const char* fname) {

	ifstream is(fname);
	if (!is) {
		cerr << "depth.cal file not found in the same folder as the depth image! " << endl;
		return false;
	}
	//read intrinsics only
	float intrinsic[9];	for (int i = 0; i<9; ++i)	is >> intrinsic[i];
	memcpy(cal, intrinsic, 9 * sizeof(float));
	is.close();

	return true;
}

void read_all_files() {

	std::string path = "E:\\Biwi\\hpdb\\";
	for (auto & p : fs::recursive_directory_iterator(path)) {
		string s = p.path().string();

		if (p.status().type() == fs::file_type::directory) {
			std::cout << p.path().string() << std::endl;
			string id = p.path().string();
			id = id.substr(id.length() - 2);

			//if (id.compare("11") != 0) continue;

			string cal_depth = p.path().string() + "\\depth.cal";
			ifstream is(cal_depth.c_str());
			if (!is) {
				cerr << "depth.cal file not found in the same folder as the depth image! " << endl;
				continue;
			}
			//read intrinsics only
			float depth_intrinsic[9];	for (int i = 0; i<9; ++i)	is >> depth_intrinsic[i];
			is.close();

			string cal_rgb = p.path().string() + "\\rgb.cal";
			ifstream is1(cal_rgb.c_str());
			if (!is) {
				cerr << "depth.cal file not found in the same folder as the depth image! " << endl;
				continue;
			}
			//read intrinsics only
			float rgb_intrinsic[9];	for (int i = 0; i<9; ++i)	is1 >> rgb_intrinsic[i];
			is1.close();

			string angles = p.path().string() +"\\angles.txt";

			std::ofstream ofs;
			ofs.open(angles, std::ofstream::out);

			ofs << "id \t frame \t roll \t pitch \t yaw \t x \t y \t z \t pixel_dx \t pixel_dy \t pixel_rx \t pixel_ry " << endl;
			for (int i = 0; i < 1000+1; i++) {

				ostringstream out;
				out << p.path().string() << "\\frame_" << std::setfill('0') << std::setw(5) << i;
				
				string pose = out.str() + +"_pose.bin";
				string rgb = out.str() + +"_rgb.png";
				string depth = out.str() + +"_depth.bin";
	
				if (std::experimental::filesystem::exists(pose)) {

					float gt[6];
					read_gt(gt, pose.c_str());

					int dx = int((gt[0] * depth_intrinsic[0]) / gt[2] + depth_intrinsic[2]);
					int dy = int((gt[1] * depth_intrinsic[4]) / gt[2] + depth_intrinsic[5]);

					int rx = int((gt[0] * rgb_intrinsic[0]) / gt[2] + rgb_intrinsic[2]);
					int ry = int((gt[1] * rgb_intrinsic[4]) / gt[2] + rgb_intrinsic[5]);
					
					//load_and_save_depth_image(depth, dx, dy);
					//load_and_save_rgb_image(rgb, rx, ry);

					// save Ground Truth data
					ofs << id << "\t";
					ofs << std::setfill('0') << std::setw(5) << i << "\t";
					ofs << std::setprecision(6);
					// roll pitch yaw
					ofs << gt[5] << "\t" << gt[3] << "\t" << gt[4] << "\t";
					ofs << gt[0] << "\t" << gt[1] << "\t" << gt[2] << "\t";
					ofs << dx << "\t" << dy << "\t" << rx << "\t" << ry;
					
					ofs << endl;
				}
			}
			ofs.close();
		}
	}
}

int main() {

	read_all_files();

	return 0;
}