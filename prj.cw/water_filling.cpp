//
// Created by danya on 20.06.2025.
//
#include "water_filling.h"

// min{input_, 0}
float inv_relu(const float input_){
	float output_;
	if (input_ > 0){
		output_=0;
	}
	else{
		output_ = input_;
	}
	return output_;
}

void downsample(cv::Mat src, cv::Mat& dst, const float rate)
{
	src.convertTo(src, CV_32F);
	const cv::Size size(0, 0);
	resize(src, dst, size, rate, rate, cv::INTER_LINEAR);
}

cv::Mat water_filling(const cv::Mat& src, const cv::Size original_size, const fs::path& path) {
	CV_Assert(src.depth() == CV_32F);

	const int height_ = src.rows;
	const int width_ = src.cols;
	const auto w_ = cv::Mat(height_, width_, CV_32F, cv::Scalar(0, 0, 0));
	auto G_ = cv::Mat(height_, width_, CV_32F, cv::Scalar(0, 0, 0));

	double G_peak; // ˆh
	double G_min;
	const auto w_ptr = reinterpret_cast<float*>(w_.data);
	const auto G_ptr = reinterpret_cast<float*>(G_.data);
	const size_t elem_step = w_.step / sizeof(float); // delta

	for (int t = 0; t < 2500; t++) {
  		G_ = w_ + src;
		cv::minMaxLoc(G_, &G_min, &G_peak);
		for (int y = 1; y < (height_ - 2); y++)
		{
			for (int x = 1; x < (width_ - 2); x++)
			{
				// hyperparameter neta
				constexpr double neta = 0.2;

				const double w_pre = w_ptr[x + y * elem_step];

				// wψ (x0, t) = (ˆh − G(x0, t)) · e−t - flooding process
				const double pouring = exp(-t) * (G_peak - G_ptr[x + y * elem_step]);

				// min{−G(x0, t) + G(x0 + ∆, t), 0} + min{− G(x0, t) + G(x0 − ∆ , t), 0}. - effusing process
				const double del_w = neta * (inv_relu(-G_ptr[x + y * elem_step] + G_ptr[x + (y + 1) * elem_step])
					+ inv_relu(-G_ptr[x + y * elem_step] + G_ptr[x + (y - 1) * elem_step])
					+ inv_relu(-G_ptr[x + y * elem_step] + G_ptr[(x + 1) + y * elem_step])
					+ inv_relu(-G_ptr[x + y * elem_step] + G_ptr[(x - 1) + y * elem_step]));

				// w(x, t) ≥ 0
				if (const float temp = del_w + pouring + w_pre; temp < 0)
				{
					w_ptr[x + y * elem_step] = 0;
				} else
				{
					w_ptr[x + y * elem_step] = temp;
				}
			}
		}

		if (t == 1500)
		{
			cv::imwrite(path.string() + "wf_t=1500.jpg", G_);
		}  else if (t == 100)
		{
			cv::imwrite(path.string() + "wf_t=100.jpg", G_);
		}
	}

	// upscale
	cv::Mat output;
	cv::resize(G_, output, original_size, 0, 0, cv::INTER_LINEAR);
	output.convertTo(output, CV_8UC1);
	return output;
}

cv::Mat incre_filling(cv::Mat input, cv::Mat Original, const fs::path& path){
	input.convertTo(input, CV_32F);
	Original.convertTo(Original, CV_32F);

	const int height = input.rows;
	const int width = input.cols;
	const auto w_ = cv::Mat(height, width, CV_32F, cv::Scalar(0, 0, 0));
	auto G_ = cv::Mat(height, width, CV_32F, cv::Scalar(0, 0, 0));

	const auto w_ptr = reinterpret_cast<float*>(w_.data);
	const auto G_ptr = reinterpret_cast<float*>(G_.data);
	const size_t elem_step = w_.step / sizeof(float);

	for (int t = 0; t < 100; t++){
		G_ = w_ + input;
		for (int y = 1; y < (height - 2); y++){

			for (int x = 1; x < (width - 2); x++){
				constexpr double neta = 0.2;

				const double w_pre = w_ptr[x + y * elem_step];
				const double del_w = neta * (-G_ptr[x + y * elem_step] + G_ptr[x + (y + 1) * elem_step]
					+ -G_ptr[x + y * elem_step] + G_ptr[x + (y - 1) * elem_step]
					+ -G_ptr[x + y * elem_step] + G_ptr[(x + 1) + y * elem_step]
					+ -G_ptr[x + y * elem_step] + G_ptr[(x - 1) + y * elem_step]);
				if (const float temp = del_w + w_pre; temp < 0){
					w_ptr[x + y*elem_step] = 0;
				}
				else{
					w_ptr[x + y*elem_step] = temp;
				}
			}
		}
		if (t == 10)
		{
			cv::imwrite(path.string() + "if_t=15.jpg", G_);
		} else if (t == 50)
		{
			cv::imwrite(path.string() + "if_t=50.jpg", G_);
		}
	}
	cv::Mat output_;

	// lim(t→∞) (I(x, y)/ G(x,y,t)) * l, l - коэффициент для изменения яркости выходного изображения, I(x, y) - оригинальное изображение
	output_ = 0.875 * Original / G_ * 255;
	output_.convertTo(output_, CV_8UC1);
	return output_;
}

cv::Mat removeShadowWaterFilling(const cv::Mat& input, float rate, const fs::path& path) {
	// Перевод из BGR в YCrCb
	cv::Mat img_YCrCb;
	cv::cvtColor(input, img_YCrCb, cv::COLOR_BGR2YCrCb);

	// Разделение на каналы: [0]=Y, [1]=Cr, [2]=Cb
	cv::Mat chan[3];
	split(img_YCrCb, chan);

	// сохранение оригинала
	cv::Mat Y = chan[0];
	const cv::Mat original_Y = chan[0].clone();

	// downsample
	downsample(Y, Y, rate);

	// Обработка яркостного канала (Y)

	// Flood and Effuse and Upscale
	cv::Mat G_ = water_filling(Y, original_Y.size(), path);

	// Incremental Filling of Catchment Basins
	G_ = incre_filling(G_, original_Y, path);

	// Объединение каналов
	std::vector<cv::Mat> channels_(3);
	channels_[0] = G_;        // Новый Y
	channels_[1] = chan[1];   // Cr
	channels_[2] = chan[2];   // Cb

	cv::Mat YCrCb_output;
	merge(channels_, YCrCb_output);

	// Обратно в BGR
	cv::Mat output;
	cv::cvtColor(YCrCb_output, output, cv::COLOR_YCrCb2BGR);

	return output;
}
