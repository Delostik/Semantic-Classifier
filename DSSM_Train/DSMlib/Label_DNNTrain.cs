using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using System.Threading.Tasks;

using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace DSMlib
{
    // Due to time limit, now only hard coded 2-class binary classification
    class Label_DNNTrain : DNN_Train
    {
        public static int MAX_SEGMENT_BATCH = 0;
        DNNRunSup dnn_runData = null;

        // for validation
        DNNRunForward dnn_forward_valid = null;
        // for test
        DNNRunForward dnn_forward_test = null;
        
        DNN dnn = null;

        double[] outputy = null;
        public double[] Ouputy
        {
            get
            {
                return outputy;
            }
        }

        LabeledInputStream trainStream = new LabeledInputStream(true);
        
        // !!!! validStream must only contain one batch, in order to be compitable with weak training
        LabeledInputStream validStream = new LabeledInputStream(false);
        LabeledInputStream testStream = new LabeledInputStream(false);


        public Label_DNNTrain()
        {
        }

        public Label_DNNTrain(DNN dnn)
        {
            this.dnn = dnn;
            //Init();
        }


        void Init()
        {
            if (dnn == null)
                throw new Exception("Must set dnn model before calling label_dnntrain init!");
            if (pairTrainFiles == null || pairTrainFiles.Count == 0)
                throw new Exception("Must set dnn model before calling label_dnntrain init!");

            //Get all maxsegsizes
            int maxread;
            for (int i = 0; i < pairTrainFiles.Count; i++)
            {
                for (int j = 0; j < pairTrainFiles[i].Count; j++)
                {
                    maxread = SequenceInputStream.get_maxSegsize(pairTrainFiles[i][j]);
                    if (maxread > MAX_SEGMENT_BATCH)
                        MAX_SEGMENT_BATCH = maxread;
                }
            }

            dnn_runData = new DNNRunSup(dnn, ParameterSetting.BATCH_SIZE, MAX_SEGMENT_BATCH);
            outputy = new double[2 * ParameterSetting.BATCH_SIZE];
            
            dnn_forward_valid = new DNNRunForward(dnn, validStream.BatchSize, validStream.MAXSEGMENT_BATCH);
            dnn_forward_test = new DNNRunForward(dnn, testStream.BatchSize, validStream.MAXSEGMENT_BATCH);
            
        }


        ~Label_DNNTrain()
        {
            Dispose();            
        }

        public override void Dispose()
        {
            trainStream.Dispose();
            validStream.Dispose();
            testStream.Dispose();
        }

        
        unsafe void calculate_outputy(float [] output, int bsize)
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
        void calculate_outputderiv(int [] labels, float [] outputderiv)
        {
            int bsize = trainStream.GPU_lbatch.batchsize;
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
        
        

        /*return the loss using by feedstream */
        //unsafe public float feedstream_batch( BatchSample_Input query_batch,  BatchSample_Input doc_batch, List<BatchSample_Input> negdoc_batches, bool train_update)
        unsafe public double feedstream_batch( LabeledBatchSample_Input batch, bool train_update)
        {
            
            
            /// forward (query doc, negdoc) streaming.
            dnn_runData.forward_activate(batch);
            dnn_runData.neurallayers.Last().Output.CopyOutFromCuda();
            calculate_outputy(dnn_runData.neurallayers.Last().Output.MemPtr, batch.batchsize);
            
            double error = 0;
            if (ParameterSetting.LOSS_REPORT == 1)
            {
                for (int i = 0; i < batch.batchsize; i++)
                {
                    double mlambda = 0;
                    mlambda = batch.Emo_Mem[i] == 1 ? Math.Log(outputy[2 * i + 1]) : Math.Log(outputy[2 * i]);
                    mlambda = -mlambda;

                    if (double.IsNaN(mlambda))
                    {
                        //Console.WriteLine("IsNaN");
                        throw new Exception("Error! NaN.");
                    }
                    if (double.IsInfinity(mlambda))
                    {
                        //Console.WriteLine("IsInfinity");
                        throw new Exception("Error! IsInfinity.");
                    }
                    error += mlambda;
                }
            }



            if (train_update)
            {
                calculate_outputderiv(batch.Emo_Mem, dnn_runData.neurallayers.Last().ErrorDeriv.MemPtr);
                dnn_runData.neurallayers.Last().ErrorDeriv.CopyIntoCuda();
                // back propagate 
                dnn_runData.backward_propagate_deriv(batch);

                // update here 
                // here we have to do all the backprop computations before updating the model, because the model's actual weights will affect the backprop computation                
                dnn_runData.update_weight(batch, LearningParameters.momentum, LearningParameters.learning_rate * batch.batchsize / ParameterSetting.BATCH_SIZE);

                if (ParameterSetting.LAST_NORMALIZATION > 0)
                {
                    dnn.neurallinks.Last().CopyOutFromCuda();
                    float[] weights = dnn.neurallinks.Last().Back_Weight;
                    float norm1 = 0, norm2 = 0;
                    for (int i = 0; i < dnn.neurallinks.Last().Neural_In.Number; i++)
                    {
                        norm1 += weights[i * 2] * weights[i * 2];
                        norm2 += weights[i * 2 + 1] * weights[i * 2 + 1];
                    }
                    norm1 = (float)Math.Sqrt(norm1);
                    norm2 = (float)Math.Sqrt(norm2);
                    if (norm1 > ParameterSetting.LAST_NORMALIZATION && norm2 > ParameterSetting.LAST_NORMALIZATION)
                    {
                        float norm_factor = Math.Max(norm1, norm2);
                        norm_factor = ParameterSetting.LAST_NORMALIZATION / norm_factor;
                        for (int j = 0; j < weights.Length; j++)
                        {
                            weights[j] = weights[j] * norm_factor;
                        }
                        dnn.neurallinks.Last().CopyIntoCuda();
                    }
                }

            }
            return error;
        }

        List<List<string>> pairTrainFiles = new List<List<string>>();
        int pairTrainFilesIdx = 0;
        List<string> ConstructShuffleTrainFiles(string file)
        {
            FileInfo fi = new FileInfo(file);
            string prefix = fi.Name.Substring(0, fi.Name.LastIndexOf('.'));
            List<string> trainFiles = new FileInfo(file).Directory.GetFiles(prefix + "_sf*").Select(o => o.FullName).ToList();
            if (File.Exists(file))
            {
                trainFiles.Add(file);
            }
            trainFiles.Sort();
            return trainFiles;
        }

        void LoadPairDataAtIdx()
        {
            Program.Print(string.Format("Loading training data : {0}",
                        new FileInfo(this.pairTrainFiles[this.pairTrainFilesIdx][0]).Name                        
                        ));
            //compose NCEProbDFile if needed
            //string nceProbFileName = null;


            trainStream.Load_Train_TriData(pairTrainFiles[pairTrainFilesIdx][0], null);
            
            //SrcNorm = Normalizer.CreateFeatureNormalize((NormalizerType)ParameterSetting.Q_FEA_NORM, TriStream.qstream.Feature_Size);
            //TgtNorm = Normalizer.CreateFeatureNormalize((NormalizerType)ParameterSetting.D_FEA_NORM, TriStream.dstream.Feature_Size);
            //TriStream.InitFeatureNorm(SrcNorm, TgtNorm);
            
            
            pairTrainFilesIdx = (pairTrainFilesIdx + 1) % pairTrainFiles.Count;            
        }

        public override void LoadTrainData(string[] files)
        {
            Program.timer.Reset();
            Program.timer.Start();

            List<string> s1TrainFiles = ConstructShuffleTrainFiles(files[0]);
            if (s1TrainFiles.Count == 0)
            {
                throw new Exception(string.Format("Error! zero training files found!"));
            }
            pairTrainFiles = Enumerable.Range(0, s1TrainFiles.Count).Select(idx => new string[] {s1TrainFiles[idx]}.ToList()).ToList();
            pairTrainFilesIdx = 0;

            LoadPairDataAtIdx();
            
            
            Program.timer.Stop();
            Program.Print("loading Training data stream done : " + Program.timer.Elapsed.ToString());
        }

        // First file is for validation, second file is for test
        public override void LoadValidateData(string[] file)
        {
            
            Program.timer.Reset();
            Program.timer.Start();
            validStream.Load_Train_TriData(file[0], null);
            testStream.Load_Train_TriData(file[1], null);
            
            //ParameterSetting.VALIDATE_QFILE, ParameterSetting.VALIDATE_DFILE, ParameterSetting.VALIDATE_QDPAIR);
            Program.timer.Stop();
            Program.Print("loading Validate stream and test stream done : " + Program.timer.Elapsed.ToString());
        }

        /// <summary>
        ///  No use now
        /// </summary>
        /// <returns></returns>
        public override float Evaluate()
        {
            return 0;
        }


        /// <summary>
        /// (not used)       
        /// </summary>
        /// <param name="srcModelPath"></param>
        /// <param name="tgtModelPath"></param>
        /// <returns></returns>
        public override float EvaluateModelOnly(string ModelPath, string extrapath = null)
        {
            return 0;
        }

        float calAccuracy(int tp, int tn, int fp, int fn)
        {
            return ((float)(tp + tn)) / ((float)(tp + tn + fn + fp));
        }
        float calF1(int tp, int tn, int fp, int fn)
        {
            float mf1pos = ((float)(2 * tp)) / ((float)(2 * tp + fp + fn));
            float mf1neg = ((float)(2 * tn)) / ((float)(2 * tn + fp + fn));
            return (mf1pos + mf1neg) / 2;
        }

        float[] evalModel()
        {
            float[] res = new float[12]; // 0-validation accuracy; 1-validation macro-f1; 2-test accuracy; 3-test macro-f1
            Program.timer.Reset();
            Program.timer.Start();
            Program.Print("Strat evaluating ...");
            validStream.Init_Batch();
            validStream.Next_Batch();
            dnn_forward_valid.forward_activate(validStream.GPU_lbatch);
            dnn_forward_valid.neurallayers.Last().Output.CopyOutFromCuda();
            float[] predictions = dnn_forward_valid.neurallayers.Last().Output.MemPtr;
            // calculate tp, tf, fp, fn
            int tp = 0, tn = 0, fp = 0, fn = 0;
            int subj_tp = 0, subj_tn = 0, subj_fp = 0, subj_fn = 0;
            int obj_tp = 0, obj_tn = 0, obj_fp = 0, obj_fn = 0;
            for (int i = 0; i < validStream.GPU_lbatch.batchsize; i++)
            {
                int l = validStream.GPU_lbatch.Emo_Mem[i];
                if (l == 1)
                {
                    if (predictions[2 * i + 1] >= predictions[2 * i])
                    {
                        tp++; // true nagtive for negtive senti
                        if (validStream.GPU_lbatch.Subj_Mem[i] == 1)
                            subj_tp++;
                        else
                            obj_tp++;
                    }
                    else
                    {
                        fn++; // false positive for negative senti
                        if (validStream.GPU_lbatch.Subj_Mem[i] == 1)
                            subj_fn++;
                        else
                            obj_fn++;
                    }
                }
                else
                {
                    if (predictions[2 * i + 1] >= predictions[2 * i])
                    {
                        fp++; // false negative for negative senti
                        if (validStream.GPU_lbatch.Subj_Mem[i] == 1)
                            subj_fp++;
                        else
                            obj_fp++;
                    }
                    else
                    {
                        tn++; // true positive for negative senti
                        if (validStream.GPU_lbatch.Subj_Mem[i] == 1)
                            subj_tn++;
                        else
                            obj_tn++;
                    }
                }
            }
            res[0] = calAccuracy(tp, tn, fp, fn);
            res[1] = calF1(tp, tn, fp, fn);

            res[4] = calAccuracy(subj_tp, subj_tn, subj_fp, subj_fn);
            res[5] = calF1(subj_tp, subj_tn, subj_fp, subj_fn);
            res[8] = calAccuracy(obj_tp, obj_tn, obj_fp, obj_fn);
            res[9] = calF1(obj_tp, obj_tn, obj_fp, obj_fn);

            testStream.Init_Batch();
            tp = 0; tn = 0; fp = 0; fn = 0;
            subj_tp = 0; subj_tn = 0; subj_fp = 0; subj_fn = 0;
            obj_tp = 0; obj_tn = 0; obj_fp = 0; obj_fn = 0;
            while (testStream.Next_Batch())
            {
                dnn_forward_test.forward_activate(testStream.GPU_lbatch);
                dnn_forward_test.neurallayers.Last().Output.CopyOutFromCuda();
                float[] pred = dnn_forward_test.neurallayers.Last().Output.MemPtr;
                for (int i = 0; i < testStream.GPU_lbatch.batchsize; i++)
                {
                    int l = testStream.GPU_lbatch.Emo_Mem[i];
                    if (l == 1)
                    {
                        if (pred[2 * i + 1] >= pred[2 * i])
                        {
                            tp++;
                            if (testStream.GPU_lbatch.Subj_Mem[i] == 1)
                                subj_tp++;
                            else
                                obj_tp++;
                        }
                        else
                        {
                            fn++;
                            if (testStream.GPU_lbatch.Subj_Mem[i] == 1)
                                subj_fn++;
                            else
                                obj_fn++;
                        }
                    }
                    else
                    {
                        if (pred[2 * i + 1] >= pred[2 * i])
                        {
                            fp++;
                            if (testStream.GPU_lbatch.Subj_Mem[i] == 1)
                                subj_fp++;
                            else
                                obj_fp++;
                        }
                        else
                        {
                            tn++;
                            if (testStream.GPU_lbatch.Subj_Mem[i] == 1)
                                subj_tn++;
                            else
                                obj_tn++;
                        }
                    }
                }
            }

            res[2] = calAccuracy(tp, tn, fp, fn);
            res[3] = calF1(tp, tn, fp, fn);

            res[6] = calAccuracy(subj_tp, subj_tn, subj_fp, subj_fn);
            res[7] = calF1(subj_tp, subj_tn, subj_fp, subj_fn);
            res[10] = calAccuracy(obj_tp, obj_tn, obj_fp, obj_fn);
            res[11] = calF1(obj_tp, obj_tn, obj_fp, obj_fn);

            Program.timer.Stop();
            Program.Print("Validation done : " + Program.timer.Elapsed.ToString());

            return res;        
        }

        void LoadModel(string queryModelFile, ref DNN model, bool allocateStructureFromEmpty)
        {
            if (allocateStructureFromEmpty)
            {
                model = new DNN(queryModelFile);
            }
            else
            {
                model.Model_Load(queryModelFile, false);
            }
            //ParameterSetting.FEATURE_DIMENSION_QUERY = queryModel.neurallayers[0].Number;
            //ParameterSetting.FEATURE_DIMENSION_DOC = docModel.neurallayers[0].Number;
        }

        public override void ModelInit_FromConfig()
        {
            if (!ParameterSetting.ISSEED)
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
                if (ParameterSetting.SUPMODEL_INIT == string.Empty || !File.Exists(ParameterSetting.SUPMODEL_INIT))
                {
                    // randomly init the whole model; if word vec init exists, use it
                    dnn.Init();
                }
                else
                {
                    // init from weak train model
                    dnn.Model_Load(ParameterSetting.SUPMODEL_INIT, false);
                    // random init last layer
                    dnn.neurallinks.Last().Init();
                }

                //ParameterSetting.FEATURE_DIMENSION_QUERY = DNN_Query.neurallayers[0].Number;
                //ParameterSetting.FEATURE_DIMENSION_DOC = DNN_Doc.neurallayers[0].Number;
            }
            else
            {
                LoadModel(ParameterSetting.SEEDMODEL1, ref dnn, true);
            }

            Program.Print("Neural Network Structure " + dnn.DNN_Descr());
            Program.Print("Math Lib " + ParameterSetting.MATH_LIB.ToString());
            if (ParameterSetting.MATH_LIB == MathLibType.cpu)
            {
                Program.Print("CPU Math thread num " + ParameterSetting.BasicMathLibThreadNum.ToString());
            }
        }
        public string ComposeDSSMModelPaths(int iter)
        {
            string ModelPath = "";
            ModelPath = ParameterSetting.MODEL_PATH + "_Sup_ITER=" + iter.ToString();
            return ModelPath;
        }

        void checkData(BatchSample_Input batch, LabeledInputStream sm, bool isTrain)
        {
            string prefix = "";
            int[] sampidx = batch.Sample_Mem;
            int[] wordidx = batch.Word_Idx_Mem;
            int[] wordMargin = batch.Seg_Margin_Mem;
            int[] feaidx = batch.Fea_Mem;
            if (sampidx[batch.batchsize - 1] > (isTrain ? MAX_SEGMENT_BATCH : sm.MAXSEGMENT_BATCH))
                Program.Print(prefix + "sample idx exceed max seg size!");
            if (sampidx[batch.batchsize - 1] != batch.elementSize)
                Program.Print(prefix + "sample idx not consistent with elementSize!");

            if (wordMargin[batch.elementSize - 1] >= batch.batchsize)
                Program.Print(prefix + "last of word margin not consistent with batch size!");

            for (int i = 0; i < batch.batchsize; i++)
            {
                if (sampidx[i] > batch.elementSize || sampidx[i] < 0)
                    Program.Print(prefix + "sample idx at position " + i.ToString() + " out of range!");
                if (i == 0)
                {
                    if (sampidx[i] < 3)
                        Program.Print(prefix + "A sentence with length < 3 found at the beginning! Length: " + sampidx[i].ToString());
                }
                else
                {
                    if (sampidx[i] - sampidx[i - 1] < 3)
                        Program.Print(prefix + "A sentence with length < 3 found at position " + i.ToString() + " ! Length: " + (sampidx[i] - sampidx[i - 1]).ToString());
                }

                if (feaidx[i] >= ParameterSetting.CONTEXT_NUM || feaidx[i] < 0)
                    Program.Print("context idx at position " + i.ToString() + " out of range!");
            }

            for (int i = 0; i < batch.elementSize; i++)
            {
                if (wordidx[i] >= ParameterSetting.WORD_NUM || wordidx[i] < 0)
                    Program.Print("Word idx at position " + i.ToString() + " out of range!");
                if (wordMargin[i] >= batch.batchsize || wordMargin[i] < 0)
                    Program.Print("Word Margin at position " + i.ToString() + " out of range!");
            }

        }

        public override void CheckDataOnly()
        {
            do
            {
                int batchidx = 0;
                trainStream.Init_Batch();
                while (trainStream.Next_Batch())
                {
                    int fileidx = pairTrainFilesIdx - 1;
                    if (fileidx < 0)
                        fileidx = pairTrainFiles.Count - 1;
                    Program.Print("Checking train file: " + pairTrainFiles[fileidx][0] + ", batchnum=" + batchidx.ToString());
                    checkData(trainStream.GPU_lbatch, trainStream, true);
                    
                    batchidx++;
                }
                LoadPairDataAtIdx();
            }
            while (pairTrainFilesIdx != 1);

            validStream.Init_Batch();
            validStream.Next_Batch();
            checkData(validStream.GPU_lbatch, validStream, false);

            testStream.Init_Batch();
            while (testStream.Next_Batch())
            {
                checkData(testStream.GPU_lbatch, testStream, false);
            }
        }

        public override void Training()
        {
            Init();
            DNN dnn_backup = null;
            Program.Print("Starting Supervised Learning!");

            double trainingLoss = 0;

            float[] previous_devEval = null;
            float[] VALIDATION_Eval = null;
            //// determin the last stopped iteration
            int lastRunStopIter = -1;
            for (int iter = 0; iter <= ParameterSetting.MAX_ITER; ++iter)
            {
                if (!File.Exists(ParameterSetting.MODEL_PATH + "_Sup_ITER=" + iter.ToString()))
                {
                    break;
                }
                lastRunStopIter = iter;                
            }

            if (lastRunStopIter == -1)
            {
                Program.Print("Initialization (Iter 0)");
                Program.Print("Saving models ...");
                dnn.CopyOutFromCuda();
                string dssmModelPath = ComposeDSSMModelPaths(0);
                dnn.Model_Save(dssmModelPath);
                
                VALIDATION_Eval = evalModel();
                Program.Print("Dataset VALIDATION :\n/*******************************/ \n" 
                                    + "Validation accuracy: " + VALIDATION_Eval[0].ToString() + "\n"
                                    + "Validation F1: " + VALIDATION_Eval[1].ToString() + "\n"
                                    + "Test accuracy: " + VALIDATION_Eval[2].ToString() + "\n"
                                    + "Test F1: " + VALIDATION_Eval[3].ToString()
                                    + "Validation accuracy(subj): " + VALIDATION_Eval[4].ToString() + "\n"
                                    + "Validation F1(subj): " + VALIDATION_Eval[5].ToString() + "\n"
                                    + "Test accuracy(subj): " + VALIDATION_Eval[6].ToString() + "\n"
                                    + "Test F1(subj): " + VALIDATION_Eval[7].ToString()
                                    + "Validation accuracy(obj): " + VALIDATION_Eval[8].ToString() + "\n"
                                    + "Validation F1(obj): " + VALIDATION_Eval[9].ToString() + "\n"
                                    + "Test accuracy(obj): " + VALIDATION_Eval[10].ToString() + "\n"
                                    + "Test F1(obj): " + VALIDATION_Eval[11].ToString()
                                    + " \n/*******************************/ \n");
                    
                File.WriteAllText(ParameterSetting.MODEL_PATH + "_Sup_LEARNING_RATE_ITER=" + 0.ToString(), LearningParameters.lr_mid.ToString());
                lastRunStopIter = 0;
            }
            else
            {
                if (ParameterSetting.VALIDATE_MODEL_ONLY)
                {
                    //// go through all previous saved runs and print validation
                    for (int iter = 0; iter <= lastRunStopIter; ++iter)
                    {
                        Program.Print("Loading from previously trained Iter " + iter.ToString());
                        string dssmModelPath = ComposeDSSMModelPaths(iter);
                        LoadModel(dssmModelPath, ref dnn, false);
                        //Program.Print("Start validation process ...");
                        VALIDATION_Eval = evalModel();
                        Program.Print("Dataset VALIDATION :\n/*******************************/ \n"
                                            + "Validation accuracy: " + VALIDATION_Eval[0].ToString() + "\n"
                                            + "Validation F1: " + VALIDATION_Eval[1].ToString() + "\n"
                                            + "Test accuracy: " + VALIDATION_Eval[2].ToString() + "\n"
                                            + "Test F1: " + VALIDATION_Eval[3].ToString()
                                            + "Validation accuracy(subj): " + VALIDATION_Eval[4].ToString() + "\n"
                                            + "Validation F1(subj): " + VALIDATION_Eval[5].ToString() + "\n"
                                            + "Test accuracy(subj): " + VALIDATION_Eval[6].ToString() + "\n"
                                            + "Test F1(subj): " + VALIDATION_Eval[7].ToString()
                                            + "Validation accuracy(obj): " + VALIDATION_Eval[8].ToString() + "\n"
                                            + "Validation F1(obj): " + VALIDATION_Eval[9].ToString() + "\n"
                                            + "Test accuracy(obj): " + VALIDATION_Eval[10].ToString() + "\n"
                                            + "Test F1(obj): " + VALIDATION_Eval[11].ToString()
                                            + " \n/*******************************/ \n"); 
                        if (File.Exists(ParameterSetting.MODEL_PATH + "_Sup_LEARNING_RATE_ITER=" + iter.ToString()))
                        {
                            LearningParameters.lr_mid = float.Parse(File.ReadAllText(ParameterSetting.MODEL_PATH + "_Sup_LEARNING_RATE_ITER=" + iter.ToString()));
                        }
                    }
                    return;
                }
                else
                {
                    //// just load the last iteration to resume
                    int iter = lastRunStopIter;
                    Program.Print("Loading from previously trained Iter " + iter.ToString());
                    string dssmModelPath = ComposeDSSMModelPaths(iter);
                    LoadModel(dssmModelPath, ref dnn, false);
                    if (File.Exists(ParameterSetting.MODEL_PATH + "_Sup_LEARNING_RATE_ITER=" + iter.ToString()))
                    {
                        LearningParameters.lr_mid = float.Parse(File.ReadAllText(ParameterSetting.MODEL_PATH + "_Sup_LEARNING_RATE_ITER=" + iter.ToString()));
                    }
                }
            }

            //// Clone to backup models
            if (ParameterSetting.ISVALIDATE && ParameterSetting.updateScheme != 2)
            {
                dnn_backup = dnn.CreateBackupClone();
            }

            
            Program.Print("total training instance number : " + trainStream.lstream.nLine.ToString());
            //Program.Print("total doc sample number : " + TriStream.dstream.total_Batch_Size.ToString());
            Program.Print("Training batches: " + trainStream.lstream.BATCH_NUM.ToString());
            Program.Print("Learning Objective : " + ParameterSetting.OBJECTIVE.ToString());
            LearningParameters.total_doc_num = trainStream.lstream.nLine;

            previous_devEval = VALIDATION_Eval;

            Program.Print("Start Training");
            Program.Print("-----------------------------------------------------------");
            int mmindex = 0;
            for (int iter = lastRunStopIter + 1; iter <= ParameterSetting.MAX_ITER; iter++)
            {
                
                
                Program.Print("ITER : " + iter.ToString());
                LearningParameters.learning_rate = LearningParameters.lr_mid;
                LearningParameters.momentum = 0.0f;

                Program.timer.Reset();
                Program.timer.Start();
                
                //// load the training file and all associated streams, the "open action" is cheap
                if (iter != lastRunStopIter + 1)
                {
                    //// we don't need to load if "iter == lastRunStopIter + 1", because it has been already opened.
                    //// we only open a new pair from the second iteration
                    LoadPairDataAtIdx();
                }

                /// adjust learning rate here.
                trainStream.Init_Batch();
                trainingLoss = 0;
                //LearningParameters.neg_static_sample = false;
                mmindex = 0;

                

                while (trainStream.Next_Batch())
                {
                    trainingLoss += feedstream_batch(trainStream.GPU_lbatch, true);
                    mmindex += 1;
                    if (mmindex % 10 == 0)
                    {
                        Console.WriteLine("Training :{0}", mmindex.ToString());
                    }
                }

                Program.Print("Training Loss : " + trainingLoss.ToString());
                Program.Print("Learning Rate : " + (LearningParameters.learning_rate.ToString()));


                dnn.CopyOutFromCuda();

                //Program.Print("Start validation process ...");
                VALIDATION_Eval = evalModel();

                Program.Print("Dataset VALIDATION :\n/*******************************/ \n"
                                            + "Validation accuracy: " + VALIDATION_Eval[0].ToString() + "\n"
                                            + "Validation F1: " + VALIDATION_Eval[1].ToString() + "\n"
                                            + "Test accuracy: " + VALIDATION_Eval[2].ToString() + "\n"
                                            + "Test F1: " + VALIDATION_Eval[3].ToString()
                                            + "Validation accuracy(subj): " + VALIDATION_Eval[4].ToString() + "\n"
                                            + "Validation F1(subj): " + VALIDATION_Eval[5].ToString() + "\n"
                                            + "Test accuracy(subj): " + VALIDATION_Eval[6].ToString() + "\n"
                                            + "Test F1(subj): " + VALIDATION_Eval[7].ToString()
                                            + "Validation accuracy(obj): " + VALIDATION_Eval[8].ToString() + "\n"
                                            + "Validation F1(obj): " + VALIDATION_Eval[9].ToString() + "\n"
                                            + "Test accuracy(obj): " + VALIDATION_Eval[10].ToString() + "\n"
                                            + "Test F1(obj): " + VALIDATION_Eval[11].ToString()
                                            + " \n/*******************************/ \n"); 


                if (ParameterSetting.updateScheme != 2)
                {
                    if (VALIDATION_Eval[0] >= previous_devEval[0] - LearningParameters.accept_range)
                    {
                        Console.WriteLine("Accepted it");
                        previous_devEval = VALIDATION_Eval;
                        if (LearningParameters.IsrateDown)
                        {
                            LearningParameters.lr_mid = LearningParameters.lr_mid * LearningParameters.down_rate;
                        }
                        //// save model to backups
                        dnn_backup.Init(dnn);
                    }
                    else
                    {
                        Console.WriteLine("Reject it");

                        LearningParameters.IsrateDown = true;
                        LearningParameters.lr_mid = LearningParameters.lr_mid * LearningParameters.reject_rate;

                        //// recover model from the last saved backup
                        dnn.Init(dnn_backup);
                    }
                }

                string dssmModelPath = ComposeDSSMModelPaths(iter);
                Program.Print("Saving models ...");
                
                dnn.Model_Save(dssmModelPath);


                //// write the learning rate after this iter
                File.WriteAllText(ParameterSetting.MODEL_PATH + "_Sup_LEARNING_RATE_ITER=" + iter.ToString(), LearningParameters.lr_mid.ToString());

                Program.timer.Stop();
                Program.Print("Training Runing Time (Iter="+ iter.ToString() +") : " + Program.timer.Elapsed.ToString());
                Program.Print("-----------------------------------------------------------");
            }

            //// Final save
            dnn.CopyOutFromCuda();
            dnn.Model_Save(ParameterSetting.MODEL_PATH + "_Sup_Final");
                        
            //pstream.General_Train_Test(ParameterSetting.TRAIN_TEST_RATE);
            //dnn_train
        }
    }
}
