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
    class Weak_DNNTrain : DNN_Train
    {
        DNNRun dnn_runData = null;
        
        DNN dnn = null;

        PairInputStream TriStream = new PairInputStream();
        
        LabeledInputStream validStream = new LabeledInputStream();

        public CudaPieceFloat distances = null;
        public IntPtr Distances {   get { return distances.CudaPtr;}   }
        public float[] Distance_Back { get { return distances.MemPtr;}   }

        //public CudaPieceFloat classWeights = null;
        //public IntPtr ClassWeights { get { return classWeights.CudaPtr; } }
        //public float[] ClassWeights_Back { get { return classWeights.MemPtr; } }
        //public int stage = 0; // 0 means the first stage (weak supervision), 1 means the second stage (supervision)


        public Weak_DNNTrain()
        {
        }

        public Weak_DNNTrain(DNN dnn)
        {
            this.dnn = dnn;
            Init();
        }


        void Init()
        {
            if (dnn == null)
                throw new Exception("Must set dnn model before calling init!");

            dnn_runData = new DNNRun(dnn);

            distances = new CudaPieceFloat(2*ParameterSetting.BATCH_SIZE, true, true);
        }


        ~Weak_DNNTrain()
        {
            Dispose();            
        }

        public override void Dispose()
        {
            if (distances != null)
                distances.Dispose();

            TriStream.Dispose();
            if (ParameterSetting.ISVALIDATE)
            {
                validStream.Dispose();
            }

        }

        unsafe void calculate_distances(int batchsize)
        {
            MathOperatorManager.GlobalInstance.Calc_EuclideanDis(dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1],
                                    dnn_runData.neurallayers.Last().Outputs[2], distances, batchsize, dnn.OutputLayerSize, ParameterSetting.DSSMEpsilon);
        }

        unsafe void calculate_outputderiv(int batchsize)
        {
            if (dnn_runData.neurallinks.Last().NeuralLinkModel.Af == A_Func.Tanh)
            {
                MathOperatorManager.GlobalInstance.Deriv_Dis(dnn_runData.neurallayers.Last().ErrorDerivs[0], dnn_runData.neurallayers.Last().ErrorDerivs[1], dnn_runData.neurallayers.Last().ErrorDerivs[2],
                                    dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1], dnn_runData.neurallayers.Last().Outputs[2], distances, batchsize, dnn.OutputLayerSize, ParameterSetting.PARM_MARGIN);
            }
            else if (dnn_runData.neurallinks.Last().NeuralLinkModel.Af == A_Func.Linear)
            {
                MathOperatorManager.GlobalInstance.Deriv_Dis_Linear(dnn_runData.neurallayers.Last().ErrorDerivs[0], dnn_runData.neurallayers.Last().ErrorDerivs[1], dnn_runData.neurallayers.Last().ErrorDerivs[2],
                                    dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1], dnn_runData.neurallayers.Last().Outputs[2], distances, batchsize, dnn.OutputLayerSize, ParameterSetting.PARM_MARGIN);
            }
            else if (dnn_runData.neurallinks.Last().NeuralLinkModel.Af == A_Func.Rectified)
            {
                MathOperatorManager.GlobalInstance.Deriv_Dis_Rectified(dnn_runData.neurallayers.Last().ErrorDerivs[0], dnn_runData.neurallayers.Last().ErrorDerivs[1], dnn_runData.neurallayers.Last().ErrorDerivs[2],
                                    dnn_runData.neurallayers.Last().Outputs[0], dnn_runData.neurallayers.Last().Outputs[1], dnn_runData.neurallayers.Last().Outputs[2], distances, batchsize, dnn.OutputLayerSize, ParameterSetting.PARM_MARGIN, ParameterSetting.DSSMEpsilon);
            }
        }
        
        unsafe void Forward_CalDistance(BatchSample_Input[] batches)
        {
            for (int q = 0; q < batches.Length; q++)
                dnn_runData.forward_activate(batches[q], q);

            calculate_distances(batches[0].batchsize);
        }

        /*return the loss using by feedstream */
        //unsafe public float feedstream_batch( BatchSample_Input query_batch,  BatchSample_Input doc_batch, List<BatchSample_Input> negdoc_batches, bool train_update)
        unsafe public float feedstream_batch( BatchSample_Input[] batches, bool train_update)
        {
            
            
            /// forward (query doc, negdoc) streaming.
            Forward_CalDistance(batches);

           
            

            float error = 0;
            if (ParameterSetting.LOSS_REPORT == 1)
            {
                distances.CopyOutFromCuda();
                for (int i = 0; i < batches[0].batchsize; i++)
                {
                    float mlambda = 0;
                    mlambda = Math.Max(0, 1 - Distance_Back[i * 2 + 1] + Distance_Back[i * 2]);

                    if (float.IsNaN(mlambda))
                    {
                        //Console.WriteLine("IsNaN");
                        throw new Exception("Error! NaN.");
                    }
                    if (float.IsInfinity(mlambda))
                    {
                        //Console.WriteLine("IsInfinity");
                        throw new Exception("Error! IsInfinity.");
                    }
                    error += mlambda;
                }
            }



            if (train_update)
            {
                calculate_outputderiv(batches[0].batchsize);


                // back propagate 
                dnn_runData.backward_propagate_deriv(batches);

                // update here 
                // here we have to do all the backprop computations before updating the model, because the model's actual weights will affect the backprop computation                
                dnn_runData.update_weight(batches, LearningParameters.momentum, LearningParameters.learning_rate * batches[0].batchsize / ParameterSetting.BATCH_SIZE);
            }
            return error;
        }

        List<List<string>> pairTrainFiles = new List<List<string>>();
        int pairTrainFilesIdx = 0;
        List<string> ConstructShuffleTrainFiles(string file)
        {
            List<string> trainFiles = new FileInfo(file).Directory.GetFiles(new FileInfo(file).Name + ".shuffle*").Select(o => o.FullName).ToList();
            if (File.Exists(file))
            {
                trainFiles.Add(file);
            }
            trainFiles.Sort();
            return trainFiles;
        }

        void LoadPairDataAtIdx()
        {
            Program.Print(string.Format("Loading training data : {0} and {1} and {2}",
                        new FileInfo(this.pairTrainFiles[this.pairTrainFilesIdx][0]).Name,
                        new FileInfo(this.pairTrainFiles[this.pairTrainFilesIdx][1]).Name,
                        new FileInfo(this.pairTrainFiles[this.pairTrainFilesIdx][2]).Name
                        ));
            //compose NCEProbDFile if needed
            //string nceProbFileName = null;


            TriStream.Load_Train_TriData(pairTrainFiles[pairTrainFilesIdx], null);
            
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
            List<string> s2TrainFiles = ConstructShuffleTrainFiles(files[1]);
            List<string> s3TrainFiles = ConstructShuffleTrainFiles(files[2]);
            if (s1TrainFiles.Count != s2TrainFiles.Count || s1TrainFiles.Count != s3TrainFiles.Count || s2TrainFiles.Count != s3TrainFiles.Count)
            {
                throw new Exception(string.Format("Error! training data have inconsistent number of training files: {0}, {1}, {2}", s1TrainFiles.Count, s2TrainFiles.Count, s3TrainFiles.Count));
            }
            if (s1TrainFiles.Count == 0)
            {
                throw new Exception(string.Format("Error! zero training files found!"));
            }
            pairTrainFiles = Enumerable.Range(0, s1TrainFiles.Count).Select(idx => new string[] {s1TrainFiles[idx], s2TrainFiles[idx], s3TrainFiles[idx]}.ToList()).ToList();
            pairTrainFilesIdx = 0;

            LoadPairDataAtIdx();
            
            
            Program.timer.Stop();
            Program.Print("loading Training data stream done : " + Program.timer.Elapsed.ToString());
        }


        public override void LoadValidateData(string[] files)
        {
            // under construction
            Program.timer.Reset();
            Program.timer.Start();
            //validStream.Load_Validate_PairData(files[0], files[1], files[2], Evaluation_Type.PairScore); 
            
            //ParameterSetting.VALIDATE_QFILE, ParameterSetting.VALIDATE_DFILE, ParameterSetting.VALIDATE_QDPAIR);
            Program.timer.Stop();
            Program.Print("loading Validate doc query stream done : " + Program.timer.Elapsed.ToString());
        }

        /// <summary>
        ///  New version. Write pair scores into a valid_score file, then call an external process to produce the metric score, and then read the metric score.
        /// </summary>
        /// <returns></returns>
        public override float Evaluate()
        {
 	        // under construction
            return 1.0f;
        }


        /// <summary>
        /// Evaluate process version 2. Don't use validation streams. But using saved models directly.        
        /// </summary>
        /// <param name="srcModelPath"></param>
        /// <param name="tgtModelPath"></param>
        /// <returns></returns>
        public override float EvaluateModelOnly(string ModelPath, string extrapath = null)
        {
            return 0;
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

                dnn.Init();

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
            ModelPath = ParameterSetting.MODEL_PATH + "_ITER=" + iter.ToString();
            return ModelPath;
        }
        public override void Training()
        {
            Init();
            DNN dnn_backup = null;
            Program.Print("Starting DNN Learning!");

            float trainingLoss = 0;

            float previous_devEval = 0;
            float VALIDATION_Eval = 0;
            //// determin the last stopped iteration
            int lastRunStopIter = -1;
            for (int iter = 0; iter <= ParameterSetting.MAX_ITER; ++iter)
            {
                if (!File.Exists(ParameterSetting.MODEL_PATH + "_ITER=" + iter.ToString()))
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
                
                if (ParameterSetting.ISVALIDATE)
                {
                    Program.Print("Start validation process ...");
                    if (!ParameterSetting.VALIDATE_MODEL_ONLY)
                    {
                        VALIDATION_Eval = Evaluate();
                    }
                    else
                    {
                        VALIDATION_Eval = EvaluateModelOnly(dssmModelPath);
                    }
                    Program.Print("Dataset VALIDATION :\n/*******************************/ \n" + VALIDATION_Eval.ToString() + " \n/*******************************/ \n");
                }
                File.WriteAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER=" + 0.ToString(), LearningParameters.lr_mid.ToString());
                lastRunStopIter = 0;
            }
            else
            {
                if (ParameterSetting.ISVALIDATE)
                {
                    //// go through all previous saved runs and print validation
                    for (int iter = 0; iter <= lastRunStopIter; ++iter)
                    {
                        Program.Print("Loading from previously trained Iter " + iter.ToString());
                        string dssmModelPath = ComposeDSSMModelPaths(iter);
                        LoadModel(dssmModelPath, ref dnn, false);
                        Program.Print("Start validation process ...");
                        if (!ParameterSetting.VALIDATE_MODEL_ONLY)
                        {
                            VALIDATION_Eval = Evaluate();
                        }
                        else
                        {
                            VALIDATION_Eval = EvaluateModelOnly(dssmModelPath);
                        }
                        Program.Print("Dataset VALIDATION :\n/*******************************/ \n" + VALIDATION_Eval.ToString() + " \n/*******************************/ \n");
                        if (File.Exists(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER=" + iter.ToString()))
                        {
                            LearningParameters.lr_mid = float.Parse(File.ReadAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER=" + iter.ToString()));
                        }
                    }
                }
                else
                {
                    //// just load the last iteration
                    int iter = lastRunStopIter;
                    Program.Print("Loading from previously trained Iter " + iter.ToString());
                    string dssmModelPath = ComposeDSSMModelPaths(iter);
                    LoadModel(dssmModelPath, ref dnn, false);
                    if (File.Exists(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER=" + iter.ToString()))
                    {
                        LearningParameters.lr_mid = float.Parse(File.ReadAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER=" + iter.ToString()));
                    }
                }
            }

            //// Clone to backup models
            if (ParameterSetting.ISVALIDATE && ParameterSetting.updateScheme != 2)
            {
                dnn_backup = dnn.CreateBackupClone();
            }

            if (ParameterSetting.NOTrain)
            {
                return;
            }
            Program.Print("total triplet instance number : " + TriStream.q0stream.nLine.ToString());
            //Program.Print("total doc sample number : " + TriStream.dstream.total_Batch_Size.ToString());
            Program.Print("Training batches: " + TriStream.q0stream.BATCH_NUM.ToString());
            Program.Print("Learning Objective : " + ParameterSetting.OBJECTIVE.ToString());
            LearningParameters.total_doc_num = TriStream.q0stream.nLine;

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
                TriStream.Init_Batch();
                trainingLoss = 0;
                //LearningParameters.neg_static_sample = false;
                mmindex = 0;

                

                while (TriStream.Next_Batch())
                {
                    trainingLoss += feedstream_batch(new BatchSample_Input[] {TriStream.GPU_q0batch, TriStream.GPU_q1batch, TriStream.GPU_q2batch}, true);
                    mmindex += 1;
                    if (mmindex % 2 == 0)
                    {
                        Console.WriteLine("Training :{0}", mmindex.ToString());
                    }
                }

                Program.Print("Training Loss : " + trainingLoss.ToString());
                Program.Print("Learning Rate : " + (LearningParameters.learning_rate.ToString()));
                
                

                if (ParameterSetting.ISVALIDATE)
                {
                    Program.Print("Start validation process ...");
                    if (!ParameterSetting.VALIDATE_MODEL_ONLY)
                    {
                        VALIDATION_Eval = Evaluate();
                    }
                    else
                    {
                        // not used
                        VALIDATION_Eval = EvaluateModelOnly("");
                    }
                    Program.Print("Dataset VALIDATION :\n/*******************************/ \n" + VALIDATION_Eval.ToString() + " \n/*******************************/ \n");

                    if (ParameterSetting.updateScheme != 2)
                    {
                        if (VALIDATION_Eval >= previous_devEval - LearningParameters.accept_range)
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
                }

                

                string dssmModelPath = ComposeDSSMModelPaths(iter);
                Program.Print("Saving models ...");
                dnn.CopyOutFromCuda();
                dnn.Model_Save(dssmModelPath);

                //// write the learning rate after this iter
                File.WriteAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER=" + iter.ToString(), LearningParameters.lr_mid.ToString());

                Program.timer.Stop();
                Program.Print("Training Runing Time (Iter="+ iter.ToString() +") : " + Program.timer.Elapsed.ToString());
                Program.Print("-----------------------------------------------------------");
            }

            //// Final save
            dnn.CopyOutFromCuda();
            dnn.Model_Save(ParameterSetting.MODEL_PATH + "_Final");
                        
            //pstream.General_Train_Test(ParameterSetting.TRAIN_TEST_RATE);
            //dnn_train
        }
    }
}
