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
            bat.elementSize = epos;
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
                                    dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1], dnn_runData.neurallayers.Last().Outputs[2], dist, batchsize, dnn.OutputLayerSize, ParameterSetting.PARM_MARGIN);
            }
            else if (dnn_runData.neurallinks.Last().NeuralLinkModel.Af == A_Func.Linear)
            {
                MathOperatorManager.GlobalInstance.Deriv_Dis_Linear(dnn_runData.neurallayers.Last().ErrorDerivs[0], dnn_runData.neurallayers.Last().ErrorDerivs[1], dnn_runData.neurallayers.Last().ErrorDerivs[2],
                                    dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1], dnn_runData.neurallayers.Last().Outputs[2], dist, batchsize, dnn.OutputLayerSize, ParameterSetting.PARM_MARGIN);
            }
            else if (dnn_runData.neurallinks.Last().NeuralLinkModel.Af == A_Func.Rectified)
            {
                MathOperatorManager.GlobalInstance.Deriv_Dis_Rectified(dnn_runData.neurallayers.Last().ErrorDerivs[0], dnn_runData.neurallayers.Last().ErrorDerivs[1], dnn_runData.neurallayers.Last().ErrorDerivs[2],
                                    dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1], dnn_runData.neurallayers.Last().Outputs[2], dist, batchsize, dnn.OutputLayerSize, ParameterSetting.PARM_MARGIN, ParameterSetting.DSSMEpsilon);
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
                ob[i] = Math.Max(0, ParameterSetting.PARM_MARGIN - dis.MemPtr[i * 2 + 1] + dis.MemPtr[i * 2]);
            }
        }

        public void CheckGrad()
        {
            Program.Print("============================================");
            Program.Print("Orig\tDelta\tComb\tGT\tError");
            generateData(fakeData);
            generateData(fakeData2);
            generateData(fakeData3);
            BatchSample_Input[] batches = new BatchSample_Input[] { fakeData, fakeData2, fakeData3 };
            //CudaPieceFloat test = new CudaPieceFloat(1, true, true);
            //Program.Print("test using gpu memory");
            Forward_CalDistance(batches, dis);
            //test.MemPtr[0] = 1;
            //test.CopyIntoCuda();
            //CudaPieceFloat test3 = new CudaPieceFloat(1, true, true);
            //Program.Print("test3 using gpu memory");
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
                        string can = objs[k].ToString() + "\t" + deltaY.ToString() + "\t" + (objs[k]+deltaY).ToString() + "\t" + newobjs[k].ToString() +"\t";
                        if (objs[k] != 0)
                            can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k]+deltaY)).ToString();
                        Program.Print(can);
                    }
                    //Program.Print("============================================");
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
                        string can = objs[k].ToString() + "\t" + deltaY.ToString() + "\t" + (objs[k] + deltaY).ToString() + "\t" + newobjs[k].ToString() + "\t";
                        if (objs[k] != 0)
                            can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k] + deltaY)).ToString();
                        Program.Print(can);
                    }
                    Program.Print("============================================");
                    bias[j] = bias[j] - CheckGradient.DELTA;
                    link.CopyIntoCuda();
                }
            }


            float[] contextlt = dnn.contextLT.Back_LookupTable;
            for (int i = 0; i < contextlt.Length; i++)
            {
                contextlt[i] = contextlt[i] + 0.001f;
                dnn.contextLT.CopyIntoCuda();
                Forward_CalDistance(batches, dis);
                calObj(newobjs);
                Program.Print("============================================");
                Program.Print("Change parameter " + i.ToString() + " in context lookup table:");
                for (int k = 0; k < objs.Length; k++)
                {
                    float deltaY = 0.001f * dnn_runData.contextLT.TabUpdate.MemPtr[i];
                    string can = objs[k].ToString() + "\t" + deltaY.ToString() + "\t" + (objs[k] + deltaY).ToString() + "\t" + newobjs[k].ToString() + "\t";
                    if (objs[k] != 0)
                        can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k] + deltaY)).ToString();
                    Program.Print(can);
                }
                Program.Print("============================================");
                contextlt[i] = contextlt[i] - 0.001f;
                dnn.contextLT.CopyIntoCuda();
            }


            float[] wordlt = dnn.wordLT.Back_LookupTable;
            //modify lookup table
            for (int i = 0; i < wordlt.Length; i++)
            {
                wordlt[i] = wordlt[i] + 0.001f;
                dnn.wordLT.CopyIntoCuda();
                Forward_CalDistance(batches, dis);
                calObj(newobjs);
                Program.Print("============================================");
                Program.Print("Change parameter " + i.ToString() + " in word lookup table:");
                for (int k = 0; k < objs.Length; k++)
                {
                    float deltaY = 0.001f * dnn_runData.wordLT.TabUpdate.MemPtr[i];
                    string can = objs[k].ToString() + "\t" + deltaY.ToString() + "\t" + (objs[k] + deltaY).ToString() + "\t" + newobjs[k].ToString() + "\t";
                    if (objs[k] != 0)
                        can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k] + deltaY)).ToString();
                    Program.Print(can);
                }
                Program.Print("============================================");
                wordlt[i] = wordlt[i] - 0.001f;
                dnn.wordLT.CopyIntoCuda();
            }
            
        }
    }

    class CheckGradientSup
    {
        DNN dnn = null;
        DNNRunSup dnn_runData = null;
        double[] outputy;
        float[] objs = new float[batchsize];
        float[] newobjs = new float[batchsize];
        public static float DELTA = 0.01f;
        public static int batchsize = 1;
        public static int maxlenSentence = 20;
        public LabeledBatchSample_Input fakeData = new LabeledBatchSample_Input(batchsize, batchsize * maxlenSentence);
        
        Random r = new Random();

        public CheckGradientSup() { }

        ~CheckGradientSup()
        {
            fakeData.Dispose();
            
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

        public void generateData(LabeledBatchSample_Input bat)
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
                bat.Emo_Mem[i] = r.Next(2);
            }
            bat.elementSize = epos;
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

            // Add the classification layer
            NeuralLayer embedlayer = dnn.neurallayers.Last();
            NeuralLayer classlayer = new NeuralLayer(2);
            dnn.neurallayers.Add(classlayer);
            dnn.neurallinks.Add(new NeuralLink(embedlayer, classlayer, A_Func.Linear, 0, ParameterSetting.LAYERWEIGHT_SIGMA.Last(), N_Type.Fully_Connected, 1, false, null, null, null));
                
            dnn.Init();
            Program.Print("Neural Network Structure " + dnn.DNN_Descr());
        }

        public void initRun()
        {
            if (dnn == null)
                throw new Exception("Must set dnn model before calling init!");

            dnn_runData = new DNNRunSup(dnn, batchsize, maxlenSentence);

            outputy = new double[2 * batchsize];
        }

        unsafe void calculate_outputy(float[] output, int bsize)
        {
            int THREAD_NUM = ParameterSetting.BasicMathLibThreadNum;
            //int bsize = trainStream.GPU_lbatch.batchsize;
            int total = bsize;

            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        double a0 = output[2 * id];
                        double a1 = output[2 * id + 1] - a0;
                        a1 = Math.Exp(a1);
                        if (double.IsPositiveInfinity(a1))
                            a1 = 1.0;
                        else
                            a1 = a1 / (1.0 + a1);
                        outputy[2 * id + 1] = a1;
                        outputy[2 * id] = 1 - a1;
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        // label 1 for positive, correspond to outputy[i*2+1] = 1, label 0 for negative, correspond to outputy[i*2] = 1
        void calculate_outputderiv(int[] labels, float[] outputderiv)
        {
            int bsize = batchsize;
            for (int i = 0; i < bsize; i++)
            {
                if (labels[i] == 1)
                {
                    outputderiv[i * 2 + 1] = (float)(outputy[i * 2 + 1] - 1);
                    outputderiv[i * 2] = (float)outputy[i * 2];
                }
                else
                {
                    outputderiv[i * 2 + 1] = (float)outputy[i * 2 + 1];
                    outputderiv[i * 2] = (float)(outputy[i * 2] - 1);
                }
            }
        }


        void calObj(float[] ob)
        {
            
            for (int i = 0; i < batchsize; i++)
            {
                ob[i] = -(float)(fakeData.Emo_Mem[i] * Math.Log(outputy[i * 2 + 1]) + (1 - fakeData.Emo_Mem[i]) * Math.Log(outputy[i * 2]));
            }
        }

        public void CheckGrad()
        {
            Program.Print("============================================");
            Program.Print("Orig\tDelta\tComb\tGT\tError");
            generateData(fakeData);
            
            
            //CudaPieceFloat test = new CudaPieceFloat(1, true, true);
            //Program.Print("test using gpu memory");
            dnn_runData.forward_activate(fakeData);
            dnn_runData.neurallayers.Last().Output.CopyOutFromCuda();
            calculate_outputy(dnn_runData.neurallayers.Last().Output.MemPtr, fakeData.batchsize);
            //test.MemPtr[0] = 1;
            //test.CopyIntoCuda();
            //CudaPieceFloat test3 = new CudaPieceFloat(1, true, true);
            //Program.Print("test3 using gpu memory");
            calObj(objs);
            calculate_outputderiv(fakeData.Emo_Mem, dnn_runData.neurallayers.Last().ErrorDeriv.MemPtr);
            dnn_runData.neurallayers.Last().ErrorDeriv.CopyIntoCuda();
            dnn_runData.backward_propagate_deriv(fakeData);

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
                NeuralLinkDataSup ldata = dnn_runData.neurallinks[i];
                float[] weight = link.Back_Weight;
                float[] bias = link.Back_Bias;
                for (int j = 0; j < weight.Length; j++)
                {
                    weight[j] = weight[j] + CheckGradient.DELTA;
                    link.CopyIntoCuda();
                    dnn_runData.forward_activate(fakeData);
                    dnn_runData.neurallayers.Last().Output.CopyOutFromCuda();
                    calculate_outputy(dnn_runData.neurallayers.Last().Output.MemPtr, fakeData.batchsize);
                    calObj(newobjs);
                    Program.Print("============================================");
                    Program.Print("Change weight " + j.ToString() + " at layer " + i.ToString() + ":");
                    for (int k = 0; k < objs.Length; k++)
                    {
                        float deltaY = CheckGradient.DELTA * ldata.WeightDeriv.MemPtr[j];
                        string can = objs[k].ToString() + "\t" + deltaY.ToString() + "\t" + (objs[k] + deltaY).ToString() + "\t" + newobjs[k].ToString() + "\t";
                        if (objs[k] != 0)
                            can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k] + deltaY)).ToString();
                        Program.Print(can);
                    }
                    //Program.Print("============================================");
                    weight[j] = weight[j] - CheckGradient.DELTA;
                    link.CopyIntoCuda();
                }

                for (int j = 0; j < bias.Length; j++)
                {
                    bias[j] = bias[j] + CheckGradient.DELTA;
                    link.CopyIntoCuda();
                    dnn_runData.forward_activate(fakeData);
                    dnn_runData.neurallayers.Last().Output.CopyOutFromCuda();
                    calculate_outputy(dnn_runData.neurallayers.Last().Output.MemPtr, fakeData.batchsize);
                    calObj(newobjs);
                    Program.Print("============================================");
                    Program.Print("Change bias " + j.ToString() + " at layer " + i.ToString() + ":");
                    for (int k = 0; k < objs.Length; k++)
                    {
                        float deltaY = CheckGradient.DELTA * ldata.BiasDeriv.MemPtr[j];
                        string can = objs[k].ToString() + "\t" + deltaY.ToString() + "\t" + (objs[k] + deltaY).ToString() + "\t" + newobjs[k].ToString() + "\t";
                        if (objs[k] != 0)
                            can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k] + deltaY)).ToString();
                        Program.Print(can);
                    }
                    Program.Print("============================================");
                    bias[j] = bias[j] - CheckGradient.DELTA;
                    link.CopyIntoCuda();
                }
            }


            float[] contextlt = dnn.contextLT.Back_LookupTable;
            for (int i = 0; i < contextlt.Length; i++)
            {
                contextlt[i] = contextlt[i] + 0.001f;
                dnn.contextLT.CopyIntoCuda();
                dnn_runData.forward_activate(fakeData);
                dnn_runData.neurallayers.Last().Output.CopyOutFromCuda();
                calculate_outputy(dnn_runData.neurallayers.Last().Output.MemPtr, fakeData.batchsize);
                calObj(newobjs);
                Program.Print("============================================");
                Program.Print("Change parameter " + i.ToString() + " in context lookup table:");
                for (int k = 0; k < objs.Length; k++)
                {
                    float deltaY = 0.001f * dnn_runData.contextLT.TabUpdate.MemPtr[i];
                    string can = objs[k].ToString() + "\t" + deltaY.ToString() + "\t" + (objs[k] + deltaY).ToString() + "\t" + newobjs[k].ToString() + "\t";
                    if (objs[k] != 0)
                        can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k] + deltaY)).ToString();
                    Program.Print(can);
                }
                Program.Print("============================================");
                contextlt[i] = contextlt[i] - 0.001f;
                dnn.contextLT.CopyIntoCuda();
            }


            float[] wordlt = dnn.wordLT.Back_LookupTable;
            //modify lookup table
            for (int i = 0; i < wordlt.Length; i++)
            {
                wordlt[i] = wordlt[i] + 0.001f;
                dnn.wordLT.CopyIntoCuda();
                dnn_runData.forward_activate(fakeData);
                dnn_runData.neurallayers.Last().Output.CopyOutFromCuda();
                calculate_outputy(dnn_runData.neurallayers.Last().Output.MemPtr, fakeData.batchsize);
                calObj(newobjs);
                Program.Print("============================================");
                Program.Print("Change parameter " + i.ToString() + " in word lookup table:");
                for (int k = 0; k < objs.Length; k++)
                {
                    float deltaY = 0.001f * dnn_runData.wordLT.TabUpdate.MemPtr[i];
                    string can = objs[k].ToString() + "\t" + deltaY.ToString() + "\t" + (objs[k] + deltaY).ToString() + "\t" + newobjs[k].ToString() + "\t";
                    if (objs[k] != 0)
                        can = can + (Math.Abs(newobjs[k] - objs[k] - deltaY) / (objs[k] + deltaY)).ToString();
                    Program.Print(can);
                }
                Program.Print("============================================");
                wordlt[i] = wordlt[i] - 0.001f;
                dnn.wordLT.CopyIntoCuda();
            }

        }
    }
}
