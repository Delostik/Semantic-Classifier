﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using System.Threading.Tasks;

using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;
namespace DSMlib
{   
    

    public class DNN_Train : IDisposable
    {
        ~DNN_Train()
        { }

        public virtual void Dispose()
        {
        }
        public virtual void LoadTrainData(string[] files)
        {  }

        public virtual void LoadValidateData(string[] files)
        { }
        
        public virtual void ModelInit_FromConfig()
        { }

        public virtual void Training()
        { }

        public virtual float Evaluate()
        {
            return 0;
        }
        public virtual float EvaluateModelOnly(string srcModelPath, string tgtModelPath)
        {
            return 0;
        }

        public virtual void CheckDataOnly()
        {

        }
    }

    //public class DSSM_Train : DNN_Train
    //{
    //    DNNRun dnn_runData = null;
        
    //    DNN dnn = null;

    //    PairInputStream PairStream = new PairInputStream();
        
    //    LabeledInputStream PairValidStream = new LabeledInputStream();

    //    public CudaPieceFloat distances = null;
    //    public IntPtr Distances {   get { return distances.CudaPtr;}   }
    //    public float[] Distance_Back { get { return distances.MemPtr;}   }

    //    public CudaPieceFloat classWeights = null;
    //    public IntPtr ClassWeights { get { return classWeights.CudaPtr; } }
    //    public float[] ClassWeights_Back { get { return classWeights.MemPtr; } }
    //    public int stage = 0; // 0 means the first stage (weak supervision), 1 means the second stage (supervision)


    //    public DSSM_Train(int stage)
    //    {
    //        this.stage = stage;
    //    }

    //    public DSSM_Train(DNN dnn, int stage)
    //    {
    //        this.dnn = dnn;
    //        this.stage = stage;
    //        Init();
    //    }


    //    void Init()
    //    {
    //        if (dnn == null)
    //            throw new Exception("Must set dnn model before calling init!");

    //        dnn_runData = new DNNRun(dnn);

    //        if (stage == 0)
    //    }


    //    ~DSSM_Train()
    //    {
    //        Dispose();            
    //    }

    //    public override void Dispose()
    //    {
    //        alphaCudaPiece.Dispose();
    //        distCudaPiece.Dispose();
    //        GPU_negative_index_Array.Dispose();
    //        GPU_Inver_negative_index_Array.Dispose();
    //        GPU_Inver_negative_value_Array.Dispose();

    //        PairStream.Dispose();
    //        if (ParameterSetting.ISVALIDATE)
    //        {
    //            PairValidStream.Dispose();
    //        }

    //        /*
    //        for (int i = 0; i < ParameterSetting.NTRIAL; i++)
    //        {
    //            GPU_negative_index[i].Dispose();
    //            GPU_Inver_negative_index[i].Dispose();
    //            GPU_Inver_negative_value[i].Dispose();
    //        }
    //        */
    //    }

    //    Random neg_random = new Random();

    //    unsafe public void calculate_deltaQD_TOP(Layer_Output_Deriv_QD_PairTOP output, int batchsize)
    //    {
    //        if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Tanh)
    //        {
    //            MathOperatorManager.GlobalInstance.Deriv_Cosine(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
    //                                output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
    //                                batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
    //        }
    //        else if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Linear)
    //        {
    //            MathOperatorManager.GlobalInstance.Derive_Cosine_Linear(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
    //                                output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
    //                                batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
    //        }
    //        else if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Rectified)
    //        {
    //            MathOperatorManager.GlobalInstance.Derive_Cosine_Rectified(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
    //                                output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
    //                                batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
    //        }
    //    }

    //    unsafe public void calculate_deltaQD_TOPEX_Full(Layer_Output_Deriv_QD_PairTOP_Full output, int batchsize)
    //    {
    //        if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Tanh)
    //        {
    //            MathOperatorManager.GlobalInstance.Deriv_Cosine_EX_Full(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
    //                    GPU_negative_index_Array, output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
    //                    ParameterSetting.NTRIAL, ParameterSetting.BATCH_SIZE, batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
    //        }
    //        else if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Linear)
    //        {
    //            MathOperatorManager.GlobalInstance.Deriv_Cosine_Linear_EX_Full(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
    //                    GPU_negative_index_Array, output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
    //                    ParameterSetting.NTRIAL, ParameterSetting.BATCH_SIZE, batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
    //        }
    //        else if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Rectified)
    //        {
    //            MathOperatorManager.GlobalInstance.Deriv_Cosine_Rectified_EX_Full(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
    //                    GPU_negative_index_Array, output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
    //                    ParameterSetting.NTRIAL, ParameterSetting.BATCH_SIZE, batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
    //        }
    //    }

    //    /* Replace by the function calculate_deltaQD_TOPEX_Full.
    //    unsafe public void calculate_deltaQD_TOPEX(int neg_index, Layer_Output_Deriv_QD_PairTOP output, int batchsize)
    //    {
    //        if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Tanh)
    //        {
    //            MathOperatorManager.GlobalInstance.Deriv_Cosine_EX(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
    //                GPU_negative_index[neg_index], output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
    //                    batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
    //        }
    //        else if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Linear)
    //        {
    //            MathOperatorManager.GlobalInstance.Derive_Cosine_Linear_EX(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
    //                GPU_negative_index[neg_index], output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
    //                    batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
    //        }
    //        else if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Rectified)
    //        {
    //            MathOperatorManager.GlobalInstance.Derive_Cosine_Rectified_EX(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
    //                GPU_negative_index[neg_index], output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
    //                    batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
    //        }
    //    }
    //    */


    //    unsafe public void Negative_Sampling(int batchSize)
    //    {
    //        for (int i = 0; i < ParameterSetting.NTRIAL; i++)
    //        {
    //            if (LearningParameters.neg_static_sample)
    //            {
    //                //int randpos = neg_random.Next((int)(0.33f * batchSize)) + (int)(0.1 * batchSize);

    //                int k1 = (int)(0.33f * batchSize);
    //                int k2 = (int)(0.67f * batchSize);
    //                int k3 = (int)(0.53f * batchSize);
    //                int k4 = (int)(0.73f * batchSize);
    //                if (k1 % batchSize == 0)
    //                {
    //                    k1 = k1 + 1;
    //                }
    //                if (k2 % batchSize == 0)
    //                {
    //                    k2 = k2 + 1;
    //                }
    //                if (k3 % batchSize == 0)
    //                {
    //                    k3 = k3 + 1;
    //                }
    //                if (k4 % batchSize == 0)
    //                {
    //                    k4 = k4 + 1;
    //                }
    //                for (int k = 0; k < batchSize; k++)
    //                {
    //                    if (i == 0)
    //                    {
    //                        GPU_negative_index_Array.MemPtr[i * ParameterSetting.BATCH_SIZE + k] = (k1 + k) % batchSize;
    //                    }
    //                    else if (i == 1)
    //                    {
    //                        GPU_negative_index_Array.MemPtr[i * ParameterSetting.BATCH_SIZE + k] = (k2 + k) % batchSize;
    //                    }
    //                    else if (i == 2)
    //                    {
    //                        GPU_negative_index_Array.MemPtr[i * ParameterSetting.BATCH_SIZE + k] = (k3 + k) % batchSize;
    //                    }
    //                    else if (i == 3)
    //                    {
    //                        GPU_negative_index_Array.MemPtr[i * ParameterSetting.BATCH_SIZE + k] = (k4 + k) % batchSize;
    //                    }
    //                }
    //            }
    //            else
    //            {
    //                int randpos = neg_random.Next((int)(0.8 * batchSize)) + (int)(0.1 * batchSize);
    //                for (int k = 0; k < batchSize; k++)
    //                {
    //                    int bs = (randpos + k) % batchSize;
    //                    GPU_negative_index_Array.MemPtr[i * ParameterSetting.BATCH_SIZE + k] = bs;
    //                }
    //            }
    //        }

    //        GPU_negative_index_Array.CopyIntoCuda();                
    //    }

    //    unsafe public void Negative_Sampling_Transpose(int batchsize)
    //    {
    //        List<List<int>> mlist = new List<List<int>>();
    //        for (int k = 0; k < batchsize; k++)
    //        {
    //            mlist.Add(new List<int>());
    //        }
    //        for (int i = 0; i < ParameterSetting.NTRIAL; i++)
    //        {
    //            for (int k = 0; k < batchsize; k++)
    //            {
    //                mlist[k].Clear();
    //            }

    //            for (int k = 0; k < batchsize; k++)
    //            {
    //                int bs = GPU_negative_index_Array.MemPtr[i * ParameterSetting.BATCH_SIZE + k]; // ].MemPtr[k];
    //                mlist[bs].Add(k);
    //            }

    //            int ptotal = 0;
    //            int pindex = 0;
    //            for (int k = 0; k < batchsize; k++)
    //            {
    //                for (int m = 0; m < mlist[k].Count; m++)
    //                {
    //                    GPU_Inver_negative_value_Array.MemPtr[i * ParameterSetting.BATCH_SIZE + pindex] = mlist[k][m]; // i].MemPtr[pindex] = mlist[k][m];
    //                    pindex++;
    //                }
    //                GPU_Inver_negative_index_Array.MemPtr[i * ParameterSetting.BATCH_SIZE + k] = ptotal + mlist[k].Count;
    //                ptotal = GPU_Inver_negative_index_Array.MemPtr[i * ParameterSetting.BATCH_SIZE + k];
    //            }
    //        }
    //        GPU_Inver_negative_index_Array.CopyIntoCuda();
    //        GPU_Inver_negative_value_Array.CopyIntoCuda();                
    //    }
        
    //    unsafe public void Forward_CalSimilarityScore(BatchSample_Input query_batch, BatchSample_Input doc_batch)
    //    {
    //        /// forward (query doc, negdoc) streaming.
    //        dnn_model_query.forward_activate(query_batch);

    //        dnn_model_doc.forward_activate(doc_batch);

    //        MathOperatorManager.GlobalInstance.Cosine_Similarity(dnn_model_query.neurallayers.Last().Output,
    //                dnn_model_doc.neurallayers.Last().Output, alphaCudaPiece, ParameterSetting.NTRIAL + 1, ParameterSetting.BATCH_SIZE, 0,
    //                query_batch.batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon); // float.Epsilon);

    //    }

    //    /*return the loss using by feedstream */
    //    //unsafe public float feedstream_batch( BatchSample_Input query_batch,  BatchSample_Input doc_batch, List<BatchSample_Input> negdoc_batches, bool train_update)
    //    unsafe public float feedstream_batch( BatchSample_Input query_batch,  BatchSample_Input doc_batch, bool train_update, StreamReader srNCEProbDist)
    //    {
    //        /// forward (query doc, negdoc) streaming.
    //        Forward_CalSimilarityScore(query_batch, doc_batch);
    //        Negative_Sampling(query_batch.batchsize);

    //        MathOperatorManager.GlobalInstance.Cosine_Similarity_EX_Full(dnn_model_query.neurallayers.Last().Output,
    //                dnn_model_doc.neurallayers.Last().Output, GPU_negative_index_Array,
    //                alphaCudaPiece, ParameterSetting.NTRIAL, ParameterSetting.BATCH_SIZE,
    //                query_batch.batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon); //float.Epsilon);
    //        /* Replace by the function Cosine_Similarity_EX_Full
    //        for (int i = 0; i < ParameterSetting.NTRIAL; i++)
    //        {    
    //            MathOperatorManager.GlobalInstance.Cosine_Similarity_EX(dnn_model_query.neurallayers.Last().Output,
    //                dnn_model_doc.neurallayers.Last().Output, GPU_negative_index[i],
    //                alphaCudaPiece, ParameterSetting.NTRIAL + 1, ParameterSetting.BATCH_SIZE, i + 1,
    //                query_batch.batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon); //float.Epsilon);
    //        }
    //        */


    //        if (ParameterSetting.OBJECTIVE == ObjectiveType.NCE)
    //        {
    //            float maxlogpD = 0;
    //            if(ParameterSetting.reserved_settings.Contains("_maxlogpd_"))
    //            {
    //                string [] spl = ParameterSetting.reserved_settings.Split('_');
    //                for(int i = 0; i < spl.Length; i++)
    //                {
    //                        if(spl[i] == "maxlogpd" && i < spl.Length-1)
    //                        {
    //                            maxlogpD = float.Parse(spl[i+1]);
    //                            break;
    //                        }
    //                }
    //            }
    //            if (srNCEProbDist != null)
    //            {
    //                string line = string.Empty;
    //                for (int i = 0; i < ParameterSetting.BATCH_SIZE; i++)
    //                {
    //                    line = srNCEProbDist.ReadLine();
    //                    if (line == null) break;
    //                    float logprob = float.Parse(line.Trim());
    //                    if (logprob > maxlogpD) logprob = maxlogpD;
    //                    doc_dist[i] = logprob;
    //                    //doc_dist[i] = -(i + 1);
    //                }
    //            }
    //            else
    //            {
    //                for (int i = 0; i < ParameterSetting.BATCH_SIZE; i++)
    //                {
    //                    doc_dist[i] = (float)(-Math.Log(LearningParameters.total_doc_num));
    //                }
    //            }

    //            distCudaPiece.CopyIntoCuda(ParameterSetting.BATCH_SIZE); //this sets D+

    //            MathOperatorManager.GlobalInstance.FillOut_Dist_NCE_Full(distCudaPiece, GPU_negative_index_Array, ParameterSetting.NTRIAL, ParameterSetting.BATCH_SIZE, doc_batch.batchsize);
    //            /// replace by the function FillOut_Dist_NCE_Full
    //            //for (int i = 0; i < ParameterSetting.NTRIAL; i++)
    //            //{ //create the [D+, D-1, D-2,...D-K] by [1, 2, ..., 1024] matrix of logP(D) -- according to Negative_Sampling and srDist, operate in GPU mem, -- hxd
    //            //    MathOperatorManager.GlobalInstance.FillOut_Dist_NCE(distCudaPiece, GPU_negative_index[i], ParameterSetting.NTRIAL + 1, ParameterSetting.BATCH_SIZE, i + 1, doc_batch.batchsize);
    //            //}
    //        }

    //        if (ParameterSetting.OBJECTIVE == ObjectiveType.MMI)
    //            MathOperatorManager.GlobalInstance.Calculate_Alpha(alphaCudaPiece, ParameterSetting.NTRIAL + 1, ParameterSetting.BATCH_SIZE, query_batch.batchsize, ParameterSetting.PARM_GAMMA);
    //        else if (ParameterSetting.OBJECTIVE == ObjectiveType.MXE)
    //            MathOperatorManager.GlobalInstance.Calculate_Alpha_MXE(alphaCudaPiece, ParameterSetting.NTRIAL + 1, ParameterSetting.BATCH_SIZE, query_batch.batchsize, ParameterSetting.PARM_GAMMA);
    //        else if (ParameterSetting.OBJECTIVE == ObjectiveType.NCE)
    //            MathOperatorManager.GlobalInstance.Calculate_Alpha_NCE(alphaCudaPiece, distCudaPiece, ParameterSetting.NTRIAL + 1, ParameterSetting.BATCH_SIZE, query_batch.batchsize, ParameterSetting.PARM_GAMMA);
    //        else if (ParameterSetting.OBJECTIVE == ObjectiveType.NCE2)
    //            MathOperatorManager.GlobalInstance.Calculate_Alpha_NCE2(alphaCudaPiece, distCudaPiece, ParameterSetting.NTRIAL + 1, ParameterSetting.BATCH_SIZE, query_batch.batchsize, ParameterSetting.PARM_GAMMA);
    //        else if (ParameterSetting.OBJECTIVE == ObjectiveType.PAIRRANK)
    //            MathOperatorManager.GlobalInstance.Calculate_Alpha_PAIRRANK(alphaCudaPiece, ParameterSetting.NTRIAL + 1, ParameterSetting.BATCH_SIZE, query_batch.batchsize, ParameterSetting.PARM_GAMMA);
    //        else throw new Exception("not supported OBJECTIVE!!");

    //        float error = 0;
    //        if (ParameterSetting.LOSS_REPORT == 1)
    //        {
    //            alphaCudaPiece.CopyOutFromCuda();
    //            for (int i = 0; i < query_batch.batchsize; i++)
    //            {
    //                float mlambda = 0;
    //                if (ParameterSetting.OBJECTIVE == ObjectiveType.MMI) //MMI L()=-logP+=log(1+sum_exp(-gamma*Delta)); first_alpha[i] = gamma*sum_exp(-gamma*Delta)/(1+sum_exp(-gamma*Delta))
    //                    mlambda = (float)Math.Log(Math.Max(float.Epsilon, (1.0 + first_alpha[i] * 1.0 / (Math.Max(ParameterSetting.PARM_GAMMA - first_alpha[i], float.Epsilon)))));

    //                else if (ParameterSetting.OBJECTIVE == ObjectiveType.MXE) //MXE L()=1-P+=1-1/(1+sum_exp(-gamma*Delta)); first_alpha[i] = gamma*sum_exp(-gamma*Delta)/(1+sum_exp(-gamma*Delta))^2
    //                    mlambda = 1 - 1 / (float)Math.Max(float.Epsilon, ((ParameterSetting.PARM_GAMMA - Math.Sqrt(ParameterSetting.PARM_GAMMA * (Math.Max(ParameterSetting.PARM_GAMMA - 4 * first_alpha[i], float.Epsilon)))) / (float.Epsilon + 2 * first_alpha[i])));

    //                else if (ParameterSetting.OBJECTIVE == ObjectiveType.NCE) //NCE
    //                {
    //                    mlambda = -(float)Math.Log(Math.Max(float.Epsilon, 1 - first_alpha[i] / ParameterSetting.PARM_GAMMA));
    //                    for (int nt = 1; nt <= ParameterSetting.NTRIAL; nt++)
    //                        mlambda += -(float)Math.Log(Math.Max(float.Epsilon, 1 - first_alpha[nt * ParameterSetting.BATCH_SIZE + i] / ParameterSetting.PARM_GAMMA));
    //                }
    //                else if (ParameterSetting.OBJECTIVE == ObjectiveType.NCE2) //NCE2
    //                {//problematic
    //                    mlambda = 1.0f - (float)((ParameterSetting.PARM_GAMMA - Math.Sqrt(ParameterSetting.PARM_GAMMA * (Math.Max(ParameterSetting.PARM_GAMMA - 4 * first_alpha[i], float.Epsilon)))) / (2 * ParameterSetting.PARM_GAMMA));
    //                    for (int nt = 1; nt <= ParameterSetting.NTRIAL; nt++)
    //                        mlambda += (float)((ParameterSetting.PARM_GAMMA - Math.Sqrt(ParameterSetting.PARM_GAMMA * (Math.Max(ParameterSetting.PARM_GAMMA - 4 * first_alpha[nt * ParameterSetting.BATCH_SIZE + i], float.Epsilon)))) / (2 * ParameterSetting.PARM_GAMMA));
    //                }
    //                else if (ParameterSetting.OBJECTIVE == ObjectiveType.PAIRRANK)
    //                {
    //                    mlambda = 0;
    //                    for (int nt = 1; nt <= ParameterSetting.NTRIAL; nt++)
    //                        mlambda += (float)(Math.Log(1.0f - first_alpha[nt * ParameterSetting.BATCH_SIZE + i] / ParameterSetting.PARM_GAMMA));
    //                }

    //                if (float.IsNaN(mlambda))
    //                {
    //                    //Console.WriteLine("IsNaN");
    //                    throw new Exception("Error! NaN.");
    //                }
    //                if (float.IsInfinity(mlambda))
    //                {
    //                    //Console.WriteLine("IsInfinity");
    //                    throw new Exception("Error! IsInfinity.");
    //                }
    //                error += mlambda;
    //            }
    //        }
    //        if (train_update)
    //        {
    //            Negative_Sampling_Transpose(query_batch.batchsize);

    //            /******* Calculate the error derivatives on the top layer outputs *****/
    //            calculate_deltaQD_TOP(Pos_QD_Pair_TOP, query_batch.batchsize);

                
    //            /// Only support GPU version now.
    //            calculate_deltaQD_TOPEX_Full(Neg_QD_Pair_TOP, query_batch.batchsize);
    //            //for (int i = 0; i < ParameterSetting.NTRIAL; i++)
    //            //{
    //            //    calculate_deltaQD_TOPEX(i, Neg_QD_Pair_TOP[i], query_batch.batchsize);
    //            //}
                
    //            // Query Derive Merge
    //            MathOperatorManager.GlobalInstance.Matrix_WeightAdd(dnn_model_query.neurallayers.Last().ErrorDeriv, Pos_QD_Pair_TOP.cuda_layer_Deriv_Q,
    //                                query_batch.batchsize, dnn_model_query.OutputLayerSize, alphaCudaPiece, 0, 0);

    //            MathOperatorManager.GlobalInstance.Matrix_WeightAdd_Full(dnn_model_query.neurallayers.Last().ErrorDeriv, Neg_QD_Pair_TOP.cuda_layer_Deriv_Q, 
    //                    ParameterSetting.NTRIAL, ParameterSetting.BATCH_SIZE, 
    //                    query_batch.batchsize,  dnn_model_query.OutputLayerSize, alphaCudaPiece, ParameterSetting.BATCH_SIZE, -1);
    //            //for (int i = 0; i < ParameterSetting.NTRIAL; i++)
    //            //{
    //            //    MathOperatorManager.GlobalInstance.Matrix_WeightAdd(dnn_model_query.neurallayers.Last().ErrorDeriv, Neg_QD_Pair_TOP[i].cuda_layer_Deriv_Q,
    //            //                    query_batch.batchsize, dnn_model_query.OutputLayerSize, alphaCudaPiece, ParameterSetting.BATCH_SIZE * (i + 1), -1);
    //            //}

    //            // Doc Derive Merge
    //            MathOperatorManager.GlobalInstance.Matrix_WeightAdd(dnn_model_doc.neurallayers.Last().ErrorDeriv, Pos_QD_Pair_TOP.cuda_layer_Deriv_D,
    //                                doc_batch.batchsize, dnn_model_doc.OutputLayerSize, alphaCudaPiece, 0, 0);

    //            MathOperatorManager.GlobalInstance.Matrix_WeightAdd_EX_Full(dnn_model_doc.neurallayers.Last().ErrorDeriv, Neg_QD_Pair_TOP.cuda_layer_Deriv_D, 
    //                                GPU_Inver_negative_index_Array,
    //                                GPU_Inver_negative_value_Array, ParameterSetting.NTRIAL, ParameterSetting.BATCH_SIZE, doc_batch.batchsize, 
    //                                dnn_model_doc.OutputLayerSize, alphaCudaPiece, ParameterSetting.BATCH_SIZE, -1);
    //            //
    //            //for (int i = 0; i < ParameterSetting.NTRIAL; i++)
    //            //{
    //            //    MathOperatorManager.GlobalInstance.Matrix_WeightAdd_EX(dnn_model_doc.neurallayers.Last().ErrorDeriv, Neg_QD_Pair_TOP[i].cuda_layer_Deriv_D, GPU_Inver_negative_index[i], GPU_Inver_negative_value[i],
    //            //                    doc_batch.batchsize, dnn_model_doc.OutputLayerSize, alphaCudaPiece, ParameterSetting.BATCH_SIZE * (i + 1), -1);
    //            //}
    //            // back propagate 
    //            dnn_model_query.backward_propagate_deriv(query_batch);
    //            dnn_model_doc.backward_propagate_deriv(doc_batch);
                
    //            // update here 
    //            // here we have to do all the backprop computations before updating the model, because the model's actual weights will affect the backprop computation                
    //            dnn_model_query.update_weight(LearningParameters.momentum, LearningParameters.learning_rate * query_batch.batchsize / ParameterSetting.BATCH_SIZE);
    //            dnn_model_doc.update_weight(LearningParameters.momentum, LearningParameters.learning_rate * query_batch.batchsize / ParameterSetting.BATCH_SIZE);

    //            // and now it should support shared models
    //        }
    //        return error;
    //    }

    //    List<Tuple<string, string>> pairTrainFiles = new List<Tuple<string, string>>();
    //    int pairTrainFilesIdx = 0;
    //    List<string> ConstructShuffleTrainFiles(string file)
    //    {
    //        List<string> trainFiles = new FileInfo(file).Directory.GetFiles(new FileInfo(file).Name + ".shuffle*").Select(o => o.FullName).ToList();
    //        if (File.Exists(file))
    //        {
    //            trainFiles.Add(file);
    //        }
    //        trainFiles.Sort();
    //        return trainFiles;
    //    }

    //    void LoadPairDataAtIdx()
    //    {
    //        Program.Print(string.Format("Loading pair training data : {0} and {1}",
    //                    new FileInfo(this.pairTrainFiles[this.pairTrainFilesIdx].Item1).Name,
    //                    new FileInfo(this.pairTrainFiles[this.pairTrainFilesIdx].Item2).Name
    //                    ));
    //        //compose NCEProbDFile if needed
    //        string nceProbFileName = null;
    //        if (ParameterSetting.OBJECTIVE == ObjectiveType.NCE) //NCE
    //        {
    //            if (!ParameterSetting.NCE_PROB_FILE.Equals("_null_"))
    //            {
    //                nceProbFileName = ParameterSetting.NCE_PROB_FILE;
    //                string tmpFileName = new FileInfo(this.pairTrainFiles[this.pairTrainFilesIdx].Item2).Name;
    //                int pos = tmpFileName.IndexOf(".shuffle");
    //                if (pos >= 0)
    //                {
    //                    nceProbFileName = ParameterSetting.NCE_PROB_FILE + tmpFileName.Substring(pos);
    //                }
    //            }
    //        }

    //        PairStream.Load_Train_PairData(pairTrainFiles[pairTrainFilesIdx].Item1, pairTrainFiles[pairTrainFilesIdx].Item2, nceProbFileName);
            
    //        SrcNorm = Normalizer.CreateFeatureNormalize((NormalizerType)ParameterSetting.Q_FEA_NORM, PairStream.qstream.Feature_Size);
    //        TgtNorm = Normalizer.CreateFeatureNormalize((NormalizerType)ParameterSetting.D_FEA_NORM, PairStream.dstream.Feature_Size);
    //        PairStream.InitFeatureNorm(SrcNorm, TgtNorm);
            
            
    //        pairTrainFilesIdx = (pairTrainFilesIdx + 1) % pairTrainFiles.Count;            
    //    }

    //    public override void LoadTrainData(string[] files)
    //    {
    //        Program.timer.Reset();
    //        Program.timer.Start();

    //        List<string> srcTrainFiles = ConstructShuffleTrainFiles(files[0]);
    //        List<string> tgtTrainFiles = ConstructShuffleTrainFiles(files[1]);
    //        if (srcTrainFiles.Count != tgtTrainFiles.Count)
    //        {
    //            throw new Exception(string.Format("Error! src and tgt have different training files: {0} vs {1}", srcTrainFiles.Count, tgtTrainFiles.Count));
    //        }
    //        if (srcTrainFiles.Count == 0)
    //        {
    //            throw new Exception(string.Format("Error! zero training files found!"));
    //        }
    //        pairTrainFiles = Enumerable.Range(0, srcTrainFiles.Count).Select(idx => new Tuple<string, string>(srcTrainFiles[idx], tgtTrainFiles[idx])).ToList();
    //        pairTrainFilesIdx = 0;

    //        LoadPairDataAtIdx();
            
            
    //        Program.timer.Stop();
    //        Program.Print("loading Training doc query stream done : " + Program.timer.Elapsed.ToString());
    //    }


    //    public override void LoadValidateData(string[] files)
    //    {
    //        Program.timer.Reset();
    //        Program.timer.Start();
    //        PairValidStream.Load_Validate_PairData(files[0], files[1], files[2], Evaluation_Type.PairScore); 
            
    //        //ParameterSetting.VALIDATE_QFILE, ParameterSetting.VALIDATE_DFILE, ParameterSetting.VALIDATE_QDPAIR);
    //        Program.timer.Stop();
    //        Program.Print("loading Validate doc query stream done : " + Program.timer.Elapsed.ToString());
    //    }

    //    /// <summary>
    //    ///  New version. Write pair scores into a valid_score file, then call an external process to produce the metric score, and then read the metric score.
    //    /// </summary>
    //    /// <returns></returns>
    //    public override float Evaluate()
    //    {
    //        PairValidStream.Init_Batch();
    //        PairValidStream.Eval_Init();
    //        while (PairValidStream.Next_Batch(SrcNorm, TgtNorm))
    //        {
    //            Forward_CalSimilarityScore(PairValidStream.qstream.Data, PairValidStream.dstream.Data);
    //            alphaCudaPiece.CopyOutFromCuda();
    //            PairValidStream.Eval_Ouput_Batch(first_alpha, null, new int[] { PairValidStream.qstream.Data.batchsize });
    //        }
            
    //        List<string> validationFileLines = null;
    //        float result = PairValidStream.Eval_Score(out validationFileLines);
            
    //        Program.Print("Validation file content :");
    //        foreach (string line in validationFileLines)
    //        {
    //            Program.Print("\t" + line);
    //        }
    //        return result;
    //    }


    //    /// <summary>
    //    /// Evaluate process version 2. Don't use validation streams. But using saved models directly.        
    //    /// </summary>
    //    /// <param name="srcModelPath"></param>
    //    /// <param name="tgtModelPath"></param>
    //    /// <returns></returns>
    //    public override float EvaluateModelOnly(string srcModelPath, string tgtModelPath)
    //    {
    //        List<string> validationFileLines = null;
    //        float result = PairValidStream.Eval_Score_ModelOnlyEvaluationModelOnly(srcModelPath, tgtModelPath, out validationFileLines);
    //        Program.Print("Validation file content :");
    //        foreach (string line in validationFileLines)
    //        {
    //            Program.Print("\t" + line);
    //        }
    //        return result;
    //    }

    //    void LoadModel(string queryModelFile, ref DNN queryModel, string docModelFile, ref DNN docModel, bool allocateStructureFromEmpty)
    //    {
    //        if (allocateStructureFromEmpty)
    //        {
    //            queryModel = new DNN(queryModelFile);
    //            if (ParameterSetting.IS_SHAREMODEL)
    //            {
    //                docModel = queryModel;
    //            }
    //            else
    //            {
    //                docModel = new DNN(docModelFile);
    //            }
    //        }
    //        else
    //        {
    //            queryModel.Model_Load(queryModelFile, false);
    //            if (ParameterSetting.IS_SHAREMODEL)
    //            {
    //                docModel = queryModel;
    //            }
    //            else
    //            {
    //                docModel.Model_Load(docModelFile, false);
    //            }
    //        }
    //        ParameterSetting.FEATURE_DIMENSION_QUERY = queryModel.neurallayers[0].Number;
    //        ParameterSetting.FEATURE_DIMENSION_DOC = docModel.neurallayers[0].Number;
    //    }

    //    public override void ModelInit_FromConfig()
    //    {
    //        if (!ParameterSetting.ISSEED)
    //        {
    //            DNN_Query = new DNN(ParameterSetting.FEATURE_DIMENSION_QUERY,
    //                ParameterSetting.SOURCE_LAYER_DIM,
    //                ParameterSetting.SOURCE_ACTIVATION,
    //                ParameterSetting.SOURCE_LAYERWEIGHT_SIGMA,
    //                ParameterSetting.SOURCE_ARCH,
    //                ParameterSetting.SOURCE_ARCH_WIND,
    //                false);

    //            if (ParameterSetting.IS_SHAREMODEL)
    //            {
    //                DNN_Doc = DNN_Query;
    //            }
    //            else
    //            {
    //                DNN_Doc = new DNN(ParameterSetting.FEATURE_DIMENSION_DOC,
    //                    ParameterSetting.TARGET_LAYER_DIM,
    //                    ParameterSetting.TARGET_ACTIVATION,
    //                    ParameterSetting.TARGET_LAYERWEIGHT_SIGMA,
    //                    ParameterSetting.TARGET_ARCH,
    //                    ParameterSetting.TARGET_ARCH_WIND,
    //                    false);
    //            }

    //            DNN_Query.Init();

    //            if (!ParameterSetting.IS_SHAREMODEL)
    //            {
    //                if (ParameterSetting.MIRROR_INIT)
    //                {
    //                    DNN_Doc.Init(DNN_Query);
    //                }
    //                else
    //                {
    //                    DNN_Doc.Init();
    //                }
    //            }
    //            ParameterSetting.FEATURE_DIMENSION_QUERY = DNN_Query.neurallayers[0].Number;
    //            ParameterSetting.FEATURE_DIMENSION_DOC = DNN_Doc.neurallayers[0].Number;
    //        }
    //        else
    //        {
    //            LoadModel(ParameterSetting.SEEDMODEL1, ref DNN_Query, ParameterSetting.SEEDMODEL2, ref DNN_Doc, true);
    //        }

    //        Program.Print("Source Neural Network Structure " + DNN_Query.DNN_Descr());
    //        Program.Print("Target Neural Network Structure " + DNN_Doc.DNN_Descr());
    //        Program.Print("Feature Num Query " + ParameterSetting.FEATURE_DIMENSION_QUERY.ToString());
    //        Program.Print("Feature Num Doc " + ParameterSetting.FEATURE_DIMENSION_DOC.ToString());
    //        Program.Print("Sharing Model " + ParameterSetting.IS_SHAREMODEL.ToString());
    //        Program.Print("Mirror Init Model " + ParameterSetting.MIRROR_INIT.ToString());
    //        Program.Print("Math Lib " + ParameterSetting.MATH_LIB.ToString());
    //        if (ParameterSetting.MATH_LIB == MathLibType.cpu)
    //        {
    //            Program.Print("CPU Math thread num " + ParameterSetting.BasicMathLibThreadNum.ToString());
    //        }
    //    }
    //    public Tuple<string, string> ComposeDSSMModelPaths(int iter)
    //    {
    //        string srcModelPath = "", tgtModelPath = "";
    //        srcModelPath = ParameterSetting.MODEL_PATH + "_QUERY_ITER" + iter.ToString();
    //        if (!ParameterSetting.IS_SHAREMODEL)
    //        {
    //            tgtModelPath = ParameterSetting.MODEL_PATH + "_DOC_ITER" + iter.ToString();
    //        }
    //        else
    //        {
    //            tgtModelPath = srcModelPath;
    //        }
    //        return new Tuple<string,string>(srcModelPath, tgtModelPath);
    //    }
    //    public override void Training()
    //    {
    //        Init(DNN_Query, DNN_Doc);
    //        DNN dnn_query_backup = null, dnn_doc_backup = null;
    //        Program.Print("Starting DNN Learning!");

    //        float trainingLoss = 0;

    //        float previous_devEval = 0;
    //        float VALIDATION_Eval = 0;
    //        //// determin the last stopped iteration
    //        int lastRunStopIter = -1;
    //        for (int iter = 0; iter <= ParameterSetting.MAX_ITER; ++iter)
    //        {
    //            if (!File.Exists(ParameterSetting.MODEL_PATH + "_QUERY_ITER" + iter.ToString()))
    //            {
    //                break;
    //            }
    //            lastRunStopIter = iter;                
    //        }

    //        if (lastRunStopIter == -1)
    //        {
    //            Program.Print("Initialization (Iter 0)");
    //            Program.Print("Saving models ...");
    //            DNN_Query.CopyOutFromCuda();
    //            Tuple<string, string> dssmModelPaths = ComposeDSSMModelPaths(0);
    //            DNN_Query.Model_Save(dssmModelPaths.Item1);
    //            if (!ParameterSetting.IS_SHAREMODEL)
    //            {
    //                DNN_Doc.CopyOutFromCuda();
    //                DNN_Doc.Model_Save(dssmModelPaths.Item2);
    //            }
    //            if (ParameterSetting.ISVALIDATE)
    //            {
    //                Program.Print("Start validation process ...");
    //                if (!ParameterSetting.VALIDATE_MODEL_ONLY)
    //                {
    //                    VALIDATION_Eval = Evaluate();
    //                }
    //                else
    //                {
    //                    VALIDATION_Eval = EvaluateModelOnly(dssmModelPaths.Item1, dssmModelPaths.Item2);
    //                }
    //                Program.Print("Dataset VALIDATION :\n/*******************************/ \n" + VALIDATION_Eval.ToString() + " \n/*******************************/ \n");
    //            }
    //            File.WriteAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER" + 0.ToString(), LearningParameters.lr_mid.ToString());
    //            lastRunStopIter = 0;
    //        }
    //        else
    //        {
    //            if (ParameterSetting.ISVALIDATE)
    //            {
    //                //// go through all previous saved runs and print validation
    //                for (int iter = 0; iter <= lastRunStopIter; ++iter)
    //                {
    //                    Program.Print("Loading from previously trained Iter " + iter.ToString());
    //                    Tuple<string, string> dssmModelPaths = ComposeDSSMModelPaths(iter);
    //                    LoadModel(dssmModelPaths.Item1,
    //                        ref DNN_Query,
    //                        dssmModelPaths.Item2,
    //                        ref DNN_Doc,
    //                        false);
    //                    Program.Print("Start validation process ...");
    //                    if (!ParameterSetting.VALIDATE_MODEL_ONLY)
    //                    {
    //                        VALIDATION_Eval = Evaluate();
    //                    }
    //                    else
    //                    {
    //                        VALIDATION_Eval = EvaluateModelOnly(dssmModelPaths.Item1, dssmModelPaths.Item2);
    //                    }
    //                    Program.Print("Dataset VALIDATION :\n/*******************************/ \n" + VALIDATION_Eval.ToString() + " \n/*******************************/ \n");
    //                    if (File.Exists(ParameterSetting.MODEL_PATH + "_LEARNING_RATE" + iter.ToString()))
    //                    {
    //                        LearningParameters.lr_mid = float.Parse(File.ReadAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE" + iter.ToString()));
    //                    }
    //                }
    //            }
    //            else
    //            {
    //                //// just load the last iteration
    //                int iter = lastRunStopIter;
    //                Program.Print("Loading from previously trained Iter " + iter.ToString());
    //                LoadModel(ParameterSetting.MODEL_PATH + "_QUERY_ITER" + iter.ToString(),
    //                    ref DNN_Query,
    //                    ParameterSetting.MODEL_PATH + "_DOC_ITER" + iter.ToString(),
    //                    ref DNN_Doc,
    //                    false);
    //                if (File.Exists(ParameterSetting.MODEL_PATH + "_LEARNING_RATE" + iter.ToString()))
    //                {
    //                    LearningParameters.lr_mid = float.Parse(File.ReadAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE" + iter.ToString()));
    //                }
    //            }
    //        }

    //        //// Clone to backup models
    //        if (ParameterSetting.ISVALIDATE)
    //        {
    //            dnn_query_backup = (DNN)DNN_Query.CreateBackupClone();
    //            if (!ParameterSetting.IS_SHAREMODEL)
    //            {
    //                dnn_doc_backup = (DNN)DNN_Doc.CreateBackupClone();
    //            }
    //        }

    //        if (ParameterSetting.NOTrain)
    //        {
    //            return;
    //        }
    //        Program.Print("total query sample number : " + PairStream.qstream.total_Batch_Size.ToString());
    //        Program.Print("total doc sample number : " + PairStream.dstream.total_Batch_Size.ToString());
    //        Program.Print("Training batches: " + PairStream.qstream.BATCH_NUM.ToString());
    //        Program.Print("Learning Objective : " + ParameterSetting.OBJECTIVE.ToString());
    //        LearningParameters.total_doc_num = PairStream.dstream.total_Batch_Size;

    //        previous_devEval = VALIDATION_Eval;

    //        Program.Print("Start Training");
    //        Program.Print("-----------------------------------------------------------");
    //        int mmindex = 0;
    //        for (int iter = lastRunStopIter + 1; iter <= ParameterSetting.MAX_ITER; iter++)
    //        {
    //            Program.Print("ITER : " + iter.ToString());
    //            LearningParameters.learning_rate = LearningParameters.lr_mid;
    //            LearningParameters.momentum = 0.0f;

    //            Program.timer.Reset();
    //            Program.timer.Start();
                
    //            //// load the training file and all associated streams, the "open action" is cheap
    //            if (iter != lastRunStopIter + 1)
    //            {
    //                //// we don't need to load if "iter == lastRunStopIter + 1", because it has been already opened.
    //                //// we only open a new pair from the second iteration
    //                LoadPairDataAtIdx();
    //            }

    //            /// adjust learning rate here.
    //            PairStream.Init_Batch();
    //            trainingLoss = 0;
    //            LearningParameters.neg_static_sample = false;
    //            mmindex = 0;                

    //            while (PairStream.Next_Batch(SrcNorm, TgtNorm))
    //            {
    //                trainingLoss += feedstream_batch(PairStream.GPU_qbatch, PairStream.GPU_dbatch, true, PairStream.srNCEProbDist);
    //                mmindex += 1;
    //                if (mmindex % 50 == 0)
    //                {
    //                    Console.Write("Training :{0}\r", mmindex.ToString());
    //                }
    //            }

    //            Program.Print("Training Loss : " + trainingLoss.ToString());
    //            Program.Print("Learning Rate : " + (LearningParameters.learning_rate.ToString()));
    //            Tuple<string, string> dssmModelPaths = ComposeDSSMModelPaths(iter);
    //            Program.Print("Saving models ...");
    //            DNN_Query.CopyOutFromCuda();
    //            DNN_Query.Model_Save(dssmModelPaths.Item1);
    //            if (!ParameterSetting.IS_SHAREMODEL)
    //            {
    //                DNN_Doc.CopyOutFromCuda();
    //                DNN_Doc.Model_Save(dssmModelPaths.Item2);
    //            }

    //            if (ParameterSetting.ISVALIDATE)
    //            {
    //                Program.Print("Start validation process ...");
    //                if (!ParameterSetting.VALIDATE_MODEL_ONLY)
    //                {
    //                    VALIDATION_Eval = Evaluate();
    //                }
    //                else
    //                {
    //                    VALIDATION_Eval = EvaluateModelOnly(dssmModelPaths.Item1, dssmModelPaths.Item2);
    //                }
    //                Program.Print("Dataset VALIDATION :\n/*******************************/ \n" + VALIDATION_Eval.ToString() + " \n/*******************************/ \n");

    //                if (VALIDATION_Eval >= previous_devEval - LearningParameters.accept_range)
    //                {
    //                    Console.WriteLine("Accepted it");
    //                    previous_devEval = VALIDATION_Eval;
    //                    if (LearningParameters.IsrateDown)
    //                    {
    //                        LearningParameters.lr_mid = LearningParameters.lr_mid * LearningParameters.down_rate;
    //                    }
    //                    //// save model to backups
    //                    dnn_query_backup.Init(DNN_Query);
    //                    if (!ParameterSetting.IS_SHAREMODEL)
    //                    {
    //                        dnn_doc_backup.Init(DNN_Doc);
    //                    }
    //                }
    //                else
    //                {
    //                    Console.WriteLine("Reject it");

    //                    LearningParameters.IsrateDown = true;
    //                    LearningParameters.lr_mid = LearningParameters.lr_mid * LearningParameters.reject_rate;

    //                    //// recover model from the last saved backup
    //                    DNN_Query.Init(dnn_query_backup);
    //                    if (!ParameterSetting.IS_SHAREMODEL)
    //                    {
    //                        DNN_Doc.Init(dnn_doc_backup);
    //                    }
    //                }
    //            }

    //            //// write the learning rate after this iter
    //            File.WriteAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER" + iter.ToString(), LearningParameters.lr_mid.ToString());

    //            Program.timer.Stop();
    //            Program.Print("Training Runing Time : " + Program.timer.Elapsed.ToString());
    //            Program.Print("-----------------------------------------------------------");
    //        }

    //        //// Final save
    //        DNN_Query.CopyOutFromCuda();
    //        DNN_Query.Model_Save(ParameterSetting.MODEL_PATH + "_QUERY_DONE");
    //        if (!ParameterSetting.IS_SHAREMODEL)
    //        {
    //            DNN_Doc.CopyOutFromCuda();
    //            DNN_Doc.Model_Save(ParameterSetting.MODEL_PATH + "_DOC_DONE");
    //        }
                        
    //        //pstream.General_Train_Test(ParameterSetting.TRAIN_TEST_RATE);
    //        //dnn_train
    //    }
        
    //}

    
}
