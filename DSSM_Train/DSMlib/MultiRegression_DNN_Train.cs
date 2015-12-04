using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DSMlib
{
    public class MultiRegression_DNN_Train : DNN_Train
    {
        public DNNRun dnn_model;
        DNN dnn = null;
        PairInputStream PairStream = new PairInputStream();
        PairInputStream PairValidStream = new PairInputStream();
        Normalizer norm = null;

        public CudaPieceFloat alphaCudaPiece = null;
        public IntPtr first_alpha_cuda { get { return alphaCudaPiece.CudaPtr; } }
        public float[] first_alpha { get { return alphaCudaPiece.MemPtr; } }

        public DNN_BatchTrain batchTrain = null;

        public MultiRegression_DNN_Train()
        { }

        public void Init(DNN m_dnn)
        {
            dnn = m_dnn;
            dnn_model = new DNNRun(dnn);
            alphaCudaPiece = new CudaPieceFloat(ParameterSetting.BATCH_SIZE * dnn_model.OutputLayerSize, true, true);

            batchTrain = new DNN_BatchTrain_CG_HS(dnn);
        }

        public MultiRegression_DNN_Train(DNN m_dnn)
        {
            // initialize the DNN model. So theoretically it supports two runs (feedforward and backward prop) on the same model instance.
            Init(m_dnn);
        }

        public void Forward_CalRegressionScore(BatchSample_Input batch)
        {
            dnn_model.forward_activate(batch);
            //return dnn_model.neurallayers.Last().Output
        }

        public override void Training()
        {
            Init(dnn);
            DNN dnn_query_backup = null;
            Program.Print("Starting DNN Learning!");
            float trainingLoss = 0;

            float previous_devEval = 0;
            float VALIDATION_Eval = 0;

            Program.Print("Initialization (Iter 0)");
            Program.Print("Saving models ...");
            dnn.CopyOutFromCuda();
            dnn.Model_Save(ParameterSetting.MODEL_PATH + "_ITER" + 0.ToString());

            if (ParameterSetting.ISVALIDATE)
            {
                dnn_query_backup = (DNN)dnn.CreateBackupClone();
                VALIDATION_Eval = Evaluate();
                Program.Print("Dataset VALIDATION :\n/*******************************/ \n" + VALIDATION_Eval.ToString() + " \n/*******************************/ \n");
            }
            File.WriteAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER" + 0.ToString(), LearningParameters.lr_mid.ToString());

            previous_devEval = VALIDATION_Eval;

            if (LearningParameters.learn_style == 1)
            {
                batchTrain.Init(dnn_model.DnnModel);
                batchTrain.LearnRate = LearningParameters.learning_rate;
            }

            Program.Print("Start Training");
            Program.Print("-----------------------------------------------------------");
            int mmindex = 0;
            for (int iter = 1; iter <= ParameterSetting.MAX_ITER; iter++)
            {
                Program.Print("ITER : " + iter.ToString());
                LearningParameters.learning_rate = LearningParameters.lr_mid;
                LearningParameters.momentum = 0f;

                Program.timer.Reset();
                Program.timer.Start();

                /// adjust learning rate here.
                PairStream.Init_Batch();
                trainingLoss = 0;

                if (LearningParameters.learn_style == 1)
                {
                    batchTrain.StartBatch();
                }

                while (PairStream.Next_Batch(norm, null))
                {
                    trainingLoss += feedstream_batch(PairStream.GPU_qbatch, PairStream.GPU_dbatch, true);
                    mmindex += 1;
                    if (mmindex % 50 == 0)
                    {
                        Program.Print("Training :{" + mmindex.ToString() + "}\r");
                    }
                }

                if (LearningParameters.learn_style == 1)
                {
                    batchTrain.EndBatch();
                }

                if (LearningParameters.learn_style == 1)
                {
                    /// batch update.
                    Program.Print("Batch Training-------------------");
                    batchTrain.Update(dnn_model.DnnModel);
                }
                else
                {
                    Program.Print("Online Training");
                }
                Program.Print("Training Loss : " + trainingLoss.ToString());
                Program.Print("Learning Rate : " + (LearningParameters.learning_rate.ToString()));

                Program.Print("Saving models ...");
                dnn.CopyOutFromCuda();
                dnn.Model_Save(ParameterSetting.MODEL_PATH + "_ITER" + iter.ToString());


                if (ParameterSetting.ISVALIDATE)
                {
                    VALIDATION_Eval = Evaluate();
                    Program.Print("Dataset VALIDATION :\n/*******************************/ \n" + VALIDATION_Eval.ToString() + " \n/*******************************/ \n");

                    float rate = (VALIDATION_Eval - previous_devEval) * 1.0f / Math.Abs(previous_devEval);


                    if (VALIDATION_Eval >= previous_devEval - LearningParameters.accept_range)
                    {
                        Program.Print("Accepted it");
                        previous_devEval = VALIDATION_Eval;
                        if (LearningParameters.IsrateDown)
                        {
                            LearningParameters.lr_mid = LearningParameters.lr_mid * LearningParameters.down_rate;
                        }

                        //// save model to backups
                        dnn_query_backup.Init(dnn);

                        /// start the batch learning mode.
                        if (rate < 0.02 && LearningParameters.learn_style == 0)
                        {
                            Program.Print("Batch Training Starting ..............................");
                            LearningParameters.learn_style = 1;
                            LearningParameters.momentum = 0;
                            batchTrain.Init(dnn_model.DnnModel);
                            batchTrain.LearnRate = LearningParameters.learning_rate;
                        }
                    }
                    else
                    {
                        Program.Print("Reject it");

                        LearningParameters.IsrateDown = true;
                        LearningParameters.lr_mid = LearningParameters.lr_mid * LearningParameters.reject_rate;

                        //// recover model from the last saved backup
                        dnn.Init(dnn_query_backup);

                        if (LearningParameters.learn_style == 1)
                        {
                            batchTrain.Init(dnn_model.DnnModel);
                            batchTrain.LearnRate = batchTrain.LearnRate / 5.0f;
                        }
                    }

                }
                //// write the learning rate after this iter
                File.WriteAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER" + iter.ToString(), LearningParameters.lr_mid.ToString());
                Program.timer.Stop();
                Program.Print("Training Runing Time : " + Program.timer.Elapsed.ToString());
                Program.Print("-----------------------------------------------------------");
            }

            //// Final save
            dnn.CopyOutFromCuda();
            dnn.Model_Save(ParameterSetting.MODEL_PATH + "_DONE");
        }

        public override void LoadTrainData(string[] files)
        {
            Program.timer.Reset();
            Program.timer.Start();
            PairStream.Load_Train_PairData(files[0], files[1]);
            norm = Normalizer.CreateFeatureNormalize((NormalizerType)ParameterSetting.Q_FEA_NORM, PairStream.qstream.Feature_Size);
            PairStream.InitFeatureNorm(norm, null);

            Program.timer.Stop();
            Program.Print("loading Training stream done : " + Program.timer.Elapsed.ToString());
        }

        public override float Evaluate()
        {
            PairValidStream.Init_Batch();
            PairValidStream.Eval_Init();
            while (PairValidStream.Next_Batch(norm, null))
            {
                ///Get the Feature batch and Label batch.
                Forward_CalRegressionScore(PairValidStream.qstream.Data);

                //, PairValidStream.dstream.Data
                dnn_model.neurallayers.Last().Output.CopyOutFromCuda();
                MathOperatorManager.GlobalInstance.Sparse2Dense_Matrix(PairValidStream.dstream.Data, alphaCudaPiece, PairValidStream.qstream.Data.batchsize, dnn_model.neurallayers.Last().Number);
                alphaCudaPiece.CopyOutFromCuda();

                PairValidStream.Eval_Ouput_Batch(dnn_model.neurallayers.Last().Output.MemPtr, alphaCudaPiece.MemPtr, new int[] { PairValidStream.qstream.Data.batchsize, dnn_model.neurallayers.Last().Number });
            }
            List<string> validationFileLines = null;
            float result = PairValidStream.Eval_Score(out validationFileLines);

            Program.Print("Validation file content :");
            foreach (string line in validationFileLines)
            {
                Program.Print("\t" + line);
            }

            return result;
        }

        public override void LoadValidateData(string[] files)
        {
            Program.timer.Reset();
            Program.timer.Start();

            PairValidStream.Load_Validate_PairData(files[0], files[1], string.Empty, Evaluation_Type.MultiRegressionScore);
            //ParameterSetting.VALIDATE_QFILE, ParameterSetting.VALIDATE_DFILE, ParameterSetting.VALIDATE_QDPAIR);
            Program.timer.Stop();
            Program.Print("loading Validate doc query stream done : " + Program.timer.Elapsed.ToString());
        }

        public override void ModelInit_FromConfig()
        {
            if (!ParameterSetting.ISSEED)
            {
                dnn = new DNN(ParameterSetting.FEATURE_DIMENSION_QUERY,
                    ParameterSetting.SOURCE_LAYER_DIM,
                    ParameterSetting.SOURCE_ACTIVATION,
                    ParameterSetting.SOURCE_LAYERWEIGHT_SIGMA,
                    ParameterSetting.SOURCE_ARCH,
                    ParameterSetting.SOURCE_ARCH_WIND,
                    false);
                dnn.Init();
            }
            else
            {
                dnn = new DNN(ParameterSetting.SEEDMODEL1);
            }
            Program.Print("Neural Network Structure " + dnn.DNN_Descr());
            Program.Print("Feature Num Query " + ParameterSetting.FEATURE_DIMENSION_QUERY.ToString());
            Program.Print("Math Lib " + ParameterSetting.MATH_LIB.ToString());
            if (ParameterSetting.MATH_LIB == MathLibType.cpu)
            {
                Program.Print("CPU Math thread num " + ParameterSetting.BasicMathLibThreadNum.ToString());
            }
        }

        unsafe public float feedstream_batch(BatchSample_Input in_batch, BatchSample_Input label_batch, bool train_update)
        {
            /// forward (query doc, negdoc) streaming.
            Forward_CalRegressionScore(in_batch);

            MathOperatorManager.GlobalInstance.Zero(alphaCudaPiece, ParameterSetting.BATCH_SIZE * dnn_model.OutputLayerSize);
            MathOperatorManager.GlobalInstance.Sparse2Dense_Matrix(label_batch, alphaCudaPiece, in_batch.batchsize, dnn_model.OutputLayerSize);

            alphaCudaPiece.CopyOutFromCuda();
            dnn_model.neurallayers.Last().Output.CopyOutFromCuda();
            MathOperatorManager.GlobalInstance.Matrix_Add(alphaCudaPiece, dnn_model.neurallayers.Last().Output, in_batch.batchsize, dnn_model.OutputLayerSize, -1);

            float error = 0;
            if (ParameterSetting.LOSS_REPORT == 1)
            {
                alphaCudaPiece.CopyOutFromCuda();
                for (int i = 0; i < in_batch.batchsize; i++)
                {
                    float mlambda = 0;
                    for (int k = 0; k < dnn_model.OutputLayerSize; k++)
                        mlambda += first_alpha[i * dnn_model.OutputLayerSize + k] * first_alpha[i * dnn_model.OutputLayerSize + k];

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
                MathOperatorManager.GlobalInstance.Zero(dnn_model.neurallayers.Last().ErrorDeriv, ParameterSetting.BATCH_SIZE * dnn_model.OutputLayerSize);

                // Query Derive Merge
                MathOperatorManager.GlobalInstance.Matrix_Add(dnn_model.neurallayers.Last().ErrorDeriv, alphaCudaPiece, in_batch.batchsize, dnn_model.OutputLayerSize, 1);

                // back propagate 
                dnn_model.backward_propagate_deriv(in_batch);

                if (LearningParameters.learn_style == 0)
                {
                    // update here 
                    // here we have to do all the backprop computations before updating the model, because the model's actual weights will affect the backprop computation                
                    dnn_model.update_weight(LearningParameters.momentum, LearningParameters.learning_rate * in_batch.batchsize / ParameterSetting.BATCH_SIZE);
                }
                else
                {
                    batchTrain.AggragateBatch(dnn_model);
                }
            }
            return error;
        }

        ~MultiRegression_DNN_Train()
        {
            Dispose();
        }

        public override void Dispose()
        {
            alphaCudaPiece.Dispose();
        }
    }
}
