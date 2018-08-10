#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <time.h>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "feat/feature-mfcc.h"
#include "ivector/voice-activity-detection.h"

#include "gmm/diag-gmm.h"
#include "hmm/posterior.h"
#include "ivector/ivector-extractor.h"
#include "ivector/plda.h"
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"
#include "ivector/agglomerative-clustering.h"
using namespace kaldi;

enum VadSource {label, energy, cnn};

void Wav2Mfcc(const WaveData wav_data, MfccOptions mfcc_opts, Matrix<BaseFloat> *mfccmat)
{
	Matrix<BaseFloat> mfccmat0;
	SubVector<BaseFloat> raw_wav(wav_data.Data(),0);
	Mfcc mfcc(mfcc_opts);
	mfcc.Compute(raw_wav, 1, &mfccmat0);
	DeltaFeaturesOptions delta_opt;
	delta_opt.order = 1;
	delta_opt.window = 3;
	ComputeDeltas(delta_opt, mfccmat0, mfccmat);
	KALDI_LOG << "[MFCC] Frame:" << mfccmat->NumRows() << " Dim:" << mfccmat->NumCols();
}

void LoadVad(std::string path, std::string name, BaseFloat length, 
	std::vector<std::pair<BaseFloat, BaseFloat>> *vad_pair)
{
	FILE* pfile = fopen(path.c_str(), "r");
	BaseFloat start, end;
	char subreco[10], reco[10];
	while(!feof(pfile))
	{
		fscanf(pfile, "%s %s %f %f", subreco, reco, &start, &end);
		std::string reco_str(reco);
		if(reco_str != name)
			continue;
		start = std::min(start*100, length);
		end = std::min(end*100, length);
		vad_pair->push_back(std::make_pair(start, end)); //[start, end)
		if(end == length)
			break;
	}
	KALDI_LOG << "[Segment] " << vad_pair->size();
}

void Mfcc2Vad(Matrix<BaseFloat> mfccmat, VadEnergyOptions vad_opts, 
	std::vector<std::pair<BaseFloat, BaseFloat>> *vad_pair)
{
	Vector<BaseFloat> vad_flags(mfccmat.NumRows());
	ComputeVadEnergy(vad_opts, mfccmat, &vad_flags);
	int start, end;
	for(int i=0; i<vad_flags.Dim(); i++)
	{
		start = i;
		while(vad_flags(i)==1.0 && i<vad_flags.Dim())
			i++;
		end = i;
		if (end > start)
			vad_pair->push_back(std::make_pair(start, end)); //[start, end)
	}
	KALDI_LOG << "[Segment] " << vad_pair->size();
}

void OverlappingVAD(std::vector<std::pair<BaseFloat, BaseFloat>> vad_pair, int max_len, 
	int shift, int min_len, std::vector<std::pair<BaseFloat, BaseFloat>> *vad_overlap)
{
	for(int i = 0; i < vad_pair.size(); i++)
	{
		int start = vad_pair[i].first, end = vad_pair[i].second;
		int start_fix = start;
		while(end - start > max_len)
		{
			vad_overlap->push_back(std::make_pair(start, start + max_len));
			start += shift;
		}
		if(end - start < min_len)
			start = std::max(end - max_len, start_fix);
		vad_overlap->push_back(std::make_pair(start, end));
	}
}

void NoOverlappingVAD(std::vector<std::pair<BaseFloat, BaseFloat>> vad_overlap, 
	std::vector<std::pair<BaseFloat, BaseFloat>> *vad_pair)
{
	vad_pair->clear();
	for(int i = 0; i < vad_overlap.size() - 1; i++)
	{
		int start = vad_overlap[i].first, end = vad_overlap[i].second;
		int next_start = vad_overlap[i+1].first;
		if(end > next_start)
		{
			end = next_start = (end + next_start) / 2;
			vad_overlap[i+1].first = next_start;
		}
		vad_pair->push_back(std::make_pair(start, end));
	}
	vad_pair->push_back(vad_overlap[vad_overlap.size()-1]);
}

void ExtractIvector(Matrix<BaseFloat> mfccmat, std::vector<std::pair<BaseFloat, BaseFloat>> vad_pair, 
	FullGmm &fgmm, IvectorExtractor extractor, Matrix<BaseFloat> *ivectormat)
{
	DiagGmm gmm;
	gmm.CopyFromFullGmm(fgmm);
	int num_gselect = 20;
	BaseFloat min_post = 0.025;
	std::vector<std::pair<BaseFloat, BaseFloat>>::iterator it;
	for(int i=0; i<vad_pair.size(); i++)
	{
		SubMatrix<BaseFloat> segment = mfccmat.RowRange(vad_pair[i].first, vad_pair[i].second - vad_pair[i].first);
		int num_frames = segment.NumRows();
		Posterior post(num_frames);
		std::vector<std::vector<int32>> gselect(segment.NumRows());
		gmm.GaussianSelection(segment, num_gselect, &gselect);
		for(int t = 0; t < num_frames; t++)
		{
			Vector<BaseFloat> loglikes;
			fgmm.LogLikelihoodsPreselect(segment.Row(t), gselect[t], &loglikes);
			loglikes.ApplySoftMax();
			for(int n = 0; n < loglikes.Dim(); n++)
			{
				if(loglikes(n) < min_post)
					loglikes(n) = 0;
			}	
			BaseFloat sum = loglikes.Sum();
			loglikes.Scale(1.0/sum);
			for(int n = 0; n < loglikes.Dim(); n++)
			{
				if(loglikes(n) != 0.0)
                	post[t].push_back(std::make_pair(gselect[t][n], loglikes(n)));
			}
		}
		bool need_2nd_order_stats = false;
		IvectorExtractorUtteranceStats utt_stats(extractor.NumGauss(), extractor.FeatDim(), need_2nd_order_stats);
		utt_stats.AccStats(segment, post);
		Vector<double> ivector(extractor.IvectorDim());
		extractor.GetIvectorDistribution(utt_stats, &ivector, NULL);
		ivector(0) -= extractor.PriorOffset();
		ivectormat->CopyRowFromVec(Vector<BaseFloat>(ivector), i);
    }
}

void NormalizeIvector(Vector<BaseFloat> mean_ivector, Matrix<BaseFloat> trans_mat, 
	Matrix<BaseFloat> ivectormat, Matrix<BaseFloat> *ivectormat_norm)
{
	for(int i=0; i<ivectormat.NumRows(); i++)
		ivectormat.Row(i).AddVec(-1, mean_ivector);

	ivectormat_norm->AddMatMat(1.0, ivectormat, kNoTrans, trans_mat, kTrans, 0.0);
	for(int i=0; i<ivectormat_norm->NumRows(); i++)
	{
		Vector<BaseFloat> ivector_norm(ivectormat_norm->Row(i));
		BaseFloat norm = ivector_norm.Norm(2.0);
		BaseFloat ratio = sqrt(ivector_norm.Dim())/norm;
		ivector_norm.Scale(ratio);
		ivectormat_norm->CopyRowFromVec(ivector_norm, i);
	}
}

bool EstPca(const Matrix<BaseFloat> &ivector_mat, BaseFloat target_energy,
  Matrix<BaseFloat> *mat) {
  int32 num_rows = ivector_mat.NumRows(),
    num_cols = ivector_mat.NumCols();
  Vector<BaseFloat> sum;
  SpMatrix<BaseFloat> sumsq;
  sum.Resize(num_cols);
  sumsq.Resize(num_cols);
  sum.AddRowSumMat(1.0, ivector_mat);
  sumsq.AddMat2(1.0, ivector_mat, kTrans, 1.0);
  sum.Scale(1.0 / num_rows);
  sumsq.Scale(1.0 / num_rows);
  sumsq.AddVec2(-1.0, sum); // now sumsq is centered covariance.
  int32 full_dim = sum.Dim();

  Matrix<BaseFloat> P(full_dim, full_dim);
  Vector<BaseFloat> s(full_dim);

  try {
    if (num_rows > num_cols)
      sumsq.Eig(&s, &P);
    else
      Matrix<BaseFloat>(sumsq).Svd(&s, &P, NULL);
  } catch (...) {
    return false;
  }

  SortSvd(&s, &P);

  Matrix<BaseFloat> transform(P, kTrans); // Transpose of P.  This is what
                                       // appears in the transform.

  // We want the PCA transform to retain target_energy amount of the total
  // energy.
  BaseFloat total_energy = s.Sum();
  BaseFloat energy = 0.0;
  int32 dim = 1;
  while (energy / total_energy <= target_energy) {
    energy += s(dim-1);
    dim++;
  }
  Matrix<BaseFloat> transform_float(transform);
  mat->Resize(transform.NumCols(), transform.NumRows());
  mat->CopyFromMat(transform);
  mat->Resize(dim, transform_float.NumCols(), kCopyData);
  return true;
}

// Transform the i-vectors using the recording-dependent PCA matrix.
void ApplyPca(const Matrix<BaseFloat> &ivectors_in,
  const Matrix<BaseFloat> &pca_mat, Matrix<BaseFloat> *ivectors_out) {
  int32 transform_cols = pca_mat.NumCols(),
        transform_rows = pca_mat.NumRows(),
        feat_dim = ivectors_in.NumCols();
  ivectors_out->Resize(ivectors_in.NumRows(), transform_rows);
  KALDI_ASSERT(transform_cols == feat_dim);
  ivectors_out->AddMatMat(1.0, ivectors_in, kNoTrans,
    pca_mat, kTrans, 0.0);
}

void PldaScore(Matrix<BaseFloat> ivectormat_pca, Plda &plda, Matrix<BaseFloat> &scoremat)
{
	PldaConfig plda_opts;
	Matrix<BaseFloat> ivectormat_plda(ivectormat_pca.NumRows(), plda.Dim());
	Vector<BaseFloat> ivector_plda(plda.Dim());
	for(int i = 0; i < ivectormat_pca.NumRows(); i++)
	{
		plda.TransformIvector(plda_opts, ivectormat_pca.Row(i), 1.0, &ivector_plda);
		ivectormat_plda.CopyRowFromVec(ivector_plda, i);
	}

	for(int i = 0; i < ivectormat_plda.NumRows(); i++)
	{
		for(int j = 0; j < ivectormat_plda.NumRows(); j++)
		{
			scoremat(i, j) = plda.LogLikelihoodRatio(
				Vector<double>(ivectormat_plda.Row(i)), 1, Vector<double>(ivectormat_plda.Row(j)));
		}
	}
}

void WriteMDTM(std::string name, std::vector<std::pair<BaseFloat, BaseFloat>> vad_pair_sp, 
	std::vector<int> spkid_sp)
{
	std::vector<int> spkid;
	std::vector<std::pair<BaseFloat, BaseFloat>> vad_pair;
	for(int i = 0; i < spkid_sp.size(); i++)
	{
		if(i == 0)
		{
			vad_pair.push_back(vad_pair_sp[i]);
			spkid.push_back(spkid_sp[i]);
			continue;
		}
		if(vad_pair_sp[i].first - vad_pair.back().second <= 1 && spkid_sp[i] == spkid.back())
			vad_pair.back().second = vad_pair_sp[i].second;
		else
		{
			vad_pair.push_back(vad_pair_sp[i]);
			spkid.push_back(spkid_sp[i]);
		}
	}
	system("mkdir -p result_mdtm");
	std::string filename = "result_mdtm/" + name + ".mdtm";
	FILE *out = fopen(filename.c_str(), "w");
	for(int i = 0; i < vad_pair.size(); i++)
	{
		fprintf(out, "%s %d %.3f %.3f %s %d\n", name.c_str(), 1, vad_pair[i].first/100.0, 
			(vad_pair[i].second - vad_pair[i].first)/100.0, "speaker", spkid[i]);
	}
	fclose(out);
}

void WriteRTTM(std::string name, std::vector<std::pair<BaseFloat, BaseFloat>> vad_pair_sp, 
	std::vector<int> spkid_sp)
{
	std::vector<int> spkid;
	std::vector<std::pair<BaseFloat, BaseFloat>> vad_pair;
	for(int i = 0; i < spkid_sp.size(); i++)
	{
		if(i == 0)
		{
			vad_pair.push_back(vad_pair_sp[i]);
			spkid.push_back(spkid_sp[i]);
			continue;
		}
		if(vad_pair_sp[i].first - vad_pair.back().second <= 1 && spkid_sp[i] == spkid.back())
			vad_pair.back().second = vad_pair_sp[i].second;
		else
		{
			vad_pair.push_back(vad_pair_sp[i]);
			spkid.push_back(spkid_sp[i]);
		}
	}
	system("mkdir -p result_rttm");
	std::string filename = "result_rttm/" + name + ".rttm";
	FILE *out = fopen(filename.c_str(), "w");
	for(int i = 0; i < vad_pair.size(); i++)
	{
		fprintf(out, "SPEAKER %s 0 %.3f %.3f <NA> <NA> %d <NA> <NA>\n", 
			name.c_str(), vad_pair[i].first/100.0, 
			(vad_pair[i].second - vad_pair[i].first)/100.0, spkid[i]);
	}
	fclose(out);
}

int main(int argc, char *argv[])
{
	MfccOptions mfcc_opts;
	mfcc_opts.num_ceps = 20;
	mfcc_opts.mel_opts.high_freq = 3700;
	mfcc_opts.mel_opts.low_freq = 20;
	mfcc_opts.frame_opts.samp_freq = 8000;
	mfcc_opts.frame_opts.frame_length_ms = 25;
	mfcc_opts.frame_opts.snip_edges = false;

	VadEnergyOptions vad_opts;
	vad_opts.vad_energy_threshold = 5;  // original: 9
	vad_opts.vad_energy_mean_scale = 0.5;
	vad_opts.vad_frames_context = 7;

	FullGmm fgmm;
	ReadKaldiObject("final.ubm", &fgmm);
	IvectorExtractor extractor;
	ReadKaldiObject("final.ie_128", &extractor);

	Matrix<BaseFloat> trans_mat;
	ReadKaldiObject("transform.mat_128_128", &trans_mat);
	Vector<BaseFloat> mean_ivector;
	ReadKaldiObject("mean.vec_128", &mean_ivector);

	Plda plda;
	ReadKaldiObject("plda_128", &plda);

	SequentialTableReader<WaveHolder> wav_reader("scp:wav.scp");
	for(; !wav_reader.Done(); wav_reader.Next())
	{
		std::string name = wav_reader.Key();
		KALDI_LOG << "Wav File: " << name;
		const WaveData &wav_data = wav_reader.Value();

		Matrix<BaseFloat> mfccmat;
		Wav2Mfcc(wav_data, mfcc_opts, &mfccmat);

		std::vector<std::pair<BaseFloat, BaseFloat>> vad_pair;
		VadSource vad_source = energy;
		switch (vad_source)
		{
			case label: LoadVad("segments", name, mfccmat.NumRows(), &vad_pair); break;
			case energy: Mfcc2Vad(mfccmat, vad_opts, &vad_pair); break;
			case cnn: LoadVad("segments_cnn", name, mfccmat.NumRows(), &vad_pair); break;
		}
		// overlapping VAD
		int max_len = 150, shift = 75, min_len = 50;
		std::vector<std::pair<BaseFloat, BaseFloat>> vad_overlap;
		OverlappingVAD(vad_pair, max_len, shift, min_len, &vad_overlap);
		NoOverlappingVAD(vad_overlap, &vad_pair);

		Matrix<BaseFloat> ivectormat(vad_overlap.size(), extractor.IvectorDim());
		ExtractIvector(mfccmat, vad_overlap, fgmm, extractor, &ivectormat);

		Matrix<BaseFloat> ivectormat_norm(ivectormat.NumRows(), trans_mat.NumRows());
		NormalizeIvector(mean_ivector, trans_mat, ivectormat, &ivectormat_norm);

		BaseFloat target_energy = 0.1;
		Matrix<BaseFloat> pca_transform;
		EstPca(ivectormat_norm, target_energy, &pca_transform);
		Matrix<BaseFloat> ivectormat_pca;
		ApplyPca(ivectormat_norm, pca_transform, &ivectormat_pca);
        
		Plda this_plda(plda);
		this_plda.ApplyTransform(Matrix<double>(pca_transform));

		Matrix<BaseFloat> scoremat(ivectormat_pca.NumRows(), ivectormat_pca.NumRows());
		PldaScore(ivectormat_pca, this_plda, scoremat);

		std::vector<int> spkid;
		BaseFloat min_cluster = 2;
		scoremat.Scale(-1);  // costs
		BaseFloat threshold = 0;
		AgglomerativeCluster(scoremat, threshold, min_cluster, &spkid);

		WriteMDTM(name, vad_pair, spkid);
		WriteRTTM(name, vad_pair, spkid);
	}
	KALDI_LOG << "[UBM] NumGauss:" << fgmm.NumGauss() << " FeatDim:" << fgmm.Dim();
	KALDI_LOG << "[Ivector] NumGauss:" << extractor.NumGauss() << " FeatDim:" 
		<< extractor.FeatDim() << " IvectorDim:" << extractor.IvectorDim();
	KALDI_LOG << "[PLDA] Dim:" << plda.Dim();
	return 0;
}
