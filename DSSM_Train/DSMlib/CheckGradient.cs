using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DSMlib
{
    class CheckGradient
    {
        DNN dnn = null;
        DNNRun dnn_runData = null;
        CudaPieceFloat dis = null;
        float[] objs = new float[batchsize];
        float[] newobjs = new float[batchsize];
        public static float DELTA = 0.01f;
        public static int batchsize = 1;
        public static int maxlenSentence = 10;
        public BatchSample_Input fakeData = new BatchSample_Input(batchsize, batchsize*maxlenSentence);
        public BatchSample_Input fakeData2 = new BatchSample_Input(batchsize, batchsize * maxlenSentence);
        public BatchSample_Input fakeData3 = new BatchSample_Input(batchsize, batchsize * maxlenSentence);
        Random r = new Random();

        public CheckGradient() {}

        ~CheckGradient()
        {
            fakeData.Dispose();
            fakeData2.Dispose();
            fakeData3.Dispose();
            dis.Dispose();
        }

        int[] randomWSequence(int size)
        {
            int[] res = new int[size];
            for (int i = 0; i < size; i++)
            {
                res[i] = r.Next(ParameterSetting.WORD_NUM);
            }
            return res;
        }

        int[] randomFeaSequence()
        {
            int[] res = new int[batchsize];
            for (int i = 0; i < batchsize; i++)
            {
                res[i] = r.Next(ParameterSetting.CONTEXT_NUM);
            }
            return res;
        }

        public void generateData(BatchSample_Input bat)
        {
            bat.batchsize = batchsize;
            int epos = 0;
            for (int i = 0; i < batchsize; i++)
            {
                int randomlen = r.Next(maxlenSentence) + 1;
                while (randomlen < 3)
                    randomlen = r.Next(maxlenSentence) + 1;
                int[] seq = randomWSequence(randomlen);
                
                for (int j = 0; j < seq.Length; j++)
                {
                    bat.Word_Idx_Mem[epos] = seq[j];
                    bat.Seg_Margin_Mem[epos] = i;
                    epos++;
                }                
                bat.Sample_Mem[i] = epos;

            }
            int[] context = randomFeaSequence();
            for (int i = 0; i < context.Length; i++)
                bat.Fea_Mem[i] = context[i];
            bat.Batch_In_GPU();
        }

        public void initDNN()
        {
            dnn = new DNN(ParameterSetting.FIXED_FEATURE_DIM,
                    ParameterSetting.LAYER_DIM,
                    ParameterSetting.ACTIVATION,
                    ParameterSetting.LAYERWEIGHT_SIGMA,
                    ParameterSetting.ARCH,
                    ParameterSetting.ARCH_WND,
                    ParameterSetting.CONTEXT_DIM,
                    ParameterSetting.WORD_NUM,
                    ParameterSetting.CONTEXT_NUM,
                    ParameterSetting.ARCH_WNDS,
                    ParameterSetting.ARCH_FMS,
                    false);
            dnn.Init();
            Program.Print("Neural Network Structure " + dnn.DNN_Descr());
        }

        public void initRun()
        {
            if (dnn == null)
                throw new Exception("Must set dnn model before calling init!");

            dnn_runData = new DNNRun(dnn);

            dis = new CudaPieceFloat(2 * batchsize, true, true);
        }

        unsafe void calculate_distances(CudaPieceFloat dist)
        {
            MathOperatorManager.GlobalInstance.Calc_EuclideanDis(dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1],
                                    dnn_runData.neurallayers.Last().Outputs[2], dist, batchsize, dnn.OutputLayerSize, ParameterSetting.DSSMEpsilon);
        }

        unsafe void calculate_outputderiv(CudaPieceFloat dist)
        {
            if (dnn_runData.neurallinks.Last().NeuralLinkModel.Af == A_Func.Tanh)
            {
                MathOperatorManager.GlobalInstance.Deriv_Dis(dnn_runData.neurallayers.Last().ErrorDerivs[0], dnn_runData.neurallayers.Last().ErrorDerivs[1], dnn_runData.neurallayers.Last().ErrorDerivs[2],
                                    dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1], dnn_runData.neurallayers.Last().Outputs[2], dist, batchsize, dnn.OutputLayerSize, 1);
            }
            else if (dnn_runData.neurallinks.Last().NeuralLinkModel.Af == A_Func.Linear)
            {
                MathOperatorManager.GlobalInstance.Deriv_Dis_Linear(dnn_runData.neurallayers.Last().ErrorDerivs[0], dnn_runData.neurallayers.Last().ErrorDerivs[1], dnn_runData.neurallayers.Last().ErrorDerivs[2],
                                    dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1], dnn_runData.neurallayers.Last().Outputs[2], dist, batchsize, dnn.OutputLayerSize, 1);
            }
            else if (dnn_runData.neurallinks.Last().NeuralLinkModel.Af == A_Func.Rectified)
            {
                MathOperatorManager.GlobalInstance.Deriv_Dis_Rectified(dnn_runData.neurallayers.Last().ErrorDerivs[0], dnn_runData.neurallayers.Last().ErrorDerivs[1], dnn_runData.neurallayers.Last().ErrorDerivs[2],
                                    dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1], dnn_runData.neurallayers.Last().Outputs[2], dist, batchsize, dnn.OutputLayerSize, 1, ParameterSetting.DSSMEpsilon);
            }
        }

        unsafe void Forward_CalDistance(BatchSample_Input[] batches, CudaPieceFloat dist)
        {
            for (int q = 0; q < batches.Length; q++)
                dnn_runData.forward_activate(batches[q], q);

            calculate_distances(dist);
        }

        void calObj(float[] ob)
        {
            dis.CopyOutFromCuda();
            for (int i = 0; i < batchsize; i++)
            {
                ob[i] = Math.Max(0, 1 - dis.MemPtr[i * 2 + 1] + dis.MemPtr[i * 2]);
            }
        }

        public void CheckGrad()
        {
            generateData(fakeData);
            generateData(fakeData2);
            generateData(fakeData3);
            BatchSample_Input[] batches = new BatchSample_Input[] { fakeData, fakeData2, fakeData3 };
            Forward_CalDistance(batches, dis);
            calObj(objs);
            calculate_outputderiv(dis);
            dnn_runData.backward_propagate_deriv(batches);

            // copy derivatives out from gpu
            for (int i = 0; i < dnn_runData.neurallinks.Count; i++)
            {
                dnn_runData.neurallinks[i].WeightDeriv.CopyOutFromCuda();
                dnn_runData.neurallinks[i].BiasDeriv.CopyOutFromCuda();
            }
            dnn_runData.wordLT.TabUpdate.CopyOutFromCuda();
            dnn_runData.contextLT.TabUpdate.CopyOutFromCuda();

            // modify each weight parameter
            for (int i = 0; i < dnn.neurallinks.Count; i++)
            {
                NeuralLink link = dnn.neurallinks[i];
                NeuralLinkData ldata = dnn_runData.neurallinks[i];
                float[] weight = link.Back_Weight;
                float[] bias = link.Back_Bias;
                for (int j = 0; j < weight.Length; j++)
                {
                    weight[j] = weight[j] + CheckGradient.DELTA;
                    link.CopyIntoCuda();
                    Forward_CalDistance(batches, dis);
                    calObj(newobjs);
                    Program.Print("============================================");
                    Program.Print("Change weight " + j.ToString() + " at layer " + i.ToString() + ":");
                    for (int k = 0; k < objs.Length; k++)
                    {
                        float deltaY = CheckGradient.DELTA * ldata.WeightDeriv.MemPtr[j];
                        string can = (objs[k]+deltaY).ToString() + "\t" + newobjs[k].ToString() +"\t";
                        if (objs[k] != 0)
                            can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k]+deltaY)).ToString();
                        Program.Print(can);
                    }
                    Program.Print("============================================");
                    weight[j] = weight[j] - CheckGradient.DELTA;
                    link.CopyIntoCuda();
                }

                for (int j = 0; j < bias.Length; j++)
                {
                    bias[j] = bias[j] + CheckGradient.DELTA;
                    link.CopyIntoCuda();
                    Forward_CalDistance(batches, dis);
                    calObj(newobjs);
                    Program.Print("============================================");
                    Program.Print("Change bias " + j.ToString() + " at layer " + i.ToString() + ":");
                    for (int k = 0; k < objs.Length; k++)
                    {
                        float deltaY = CheckGradient.DELTA * ldata.BiasDeriv.MemPtr[j];
                        string can = (objs[k] + deltaY).ToString() + "\t" + newobjs[k].ToString() + "\t";
                        if (objs[k] != 0)
                            can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k] + deltaY)).ToString();
                        Program.Print(can);
                    }
                    Program.Print("============================================");
                    bias[j] = bias[j] - CheckGradient.DELTA;
                    link.CopyIntoCuda();
                }
            }

            float[] wordlt = dnn.wordLT.Back_LookupTable;
            //modify lookup table
            for (int i = 0; i < wordlt.Length; i++)
            {
                wordlt[i] = wordlt[i] + CheckGradient.DELTA;
                dnn.wordLT.CopyIntoCuda();
                Forward_CalDistance(batches, dis);
                calObj(newobjs);
                Program.Print("============================================");
                Program.Print("Change parameter " + i.ToString() + " in word lookup table:");
                for (int k = 0; k < objs.Length; k++)
                {
                    float deltaY = CheckGradient.DELTA * dnn_runData.wordLT.TabUpdate.MemPtr[i];
                    string can = (objs[k] + deltaY).ToString() + "\t" + newobjs[k].ToString() + "\t";
                    if (objs[k] != 0)
                        can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k] + deltaY)).ToString();
                    Program.Print(can);
                }
                Program.Print("============================================");
                wordlt[i] = wordlt[i] - CheckGradient.DELTA;
                dnn.wordLT.CopyIntoCuda();
            }
            float[] contextlt = dnn.contextLT.Back_LookupTable;
            for (int i = 0; i < contextlt.Length; i++)
            {
                contextlt[i] = contextlt[i] + CheckGradient.DELTA;
                dnn.contextLT.CopyIntoCuda();
                Forward_CalDistance(batches, dis);
                calObj(newobjs);
                Program.Print("============================================");
                Program.Print("Change parameter " + i.ToString() + " in context lookup table:");
                for (int k = 0; k < objs.Length; k++)
                {
                    float deltaY = CheckGradient.DELTA * dnn_runData.contextLT.TabUpdate.MemPtr[i];
                    string can = (objs[k] + deltaY).ToString() + "\t" + newobjs[k].ToString() + "\t";
                    if (objs[k] != 0)
                        can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k] + deltaY)).ToString();
                    Program.Print(can);
                }
                Program.Print("============================================");
                contextlt[i] = contextlt[i] - CheckGradient.DELTA;
                dnn.contextLT.CopyIntoCuda();
            }
        }
    }
}
