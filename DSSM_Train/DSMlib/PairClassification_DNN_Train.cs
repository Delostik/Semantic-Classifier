using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DSMlib
{
    public class PairClassification_DNN_Train : DNN_Train
    {
        PairInputStream PairStream = new PairInputStream();
        PairInputStream PairValidStream = new PairInputStream();
        float[] Label = null;
        int LabelDim = 0;
        int SubspaceDim = 0;
        int LabelCursor = 0;

        Normalizer SrcNorm = null;
        Normalizer TgtNorm = null;

        DNN DNN_Query = null;
        DNN DNN_Doc = null;

        DNNRun dnn_model_query;
        DNNRun dnn_model_doc;

        public CudaPieceFloat alphaCudaPiece = null;
        public CudaPieceFloat betaCudaPiece = null;
        public PairClassification_DNN_Train()
        {
        }

        void LoadModel(string queryModelFile, ref DNN queryModel, string docModelFile, ref DNN docModel, bool allocateStructureFromEmpty)
        {
            if (allocateStructureFromEmpty)
            {
                queryModel = new DNN(queryModelFile);
                if (ParameterSetting.IS_SHAREMODEL)
                {
                    docModel = queryModel;
                }
                else
                {
                    docModel = new DNN(docModelFile);
                }
            }
            else
            {
                queryModel.Model_Load(queryModelFile, false);
                if (ParameterSetting.IS_SHAREMODEL)
                {
                    docModel = queryModel;
                }
                else
                {
                    docModel.Model_Load(docModelFile, false);
                }
            }
            ParameterSetting.FEATURE_DIMENSION_QUERY = queryModel.neurallayers[0].Number;
            ParameterSetting.FEATURE_DIMENSION_DOC = docModel.neurallayers[0].Number;
        }

        public override void ModelInit_FromConfig()
        {
            int src_layer = ParameterSetting.SOURCE_LAYER_DIM.Length - 1;

            SubspaceDim = ParameterSetting.SOURCE_LAYER_DIM[src_layer];
            ParameterSetting.SOURCE_LAYER_DIM[src_layer] = ParameterSetting.SOURCE_LAYER_DIM[src_layer] * LabelDim;

            int tgt_layer = ParameterSetting.TARGET_LAYER_DIM.Length - 1;
            ParameterSetting.TARGET_LAYER_DIM[tgt_layer] = ParameterSetting.TARGET_LAYER_DIM[tgt_layer] * LabelDim;

            if (!ParameterSetting.ISSEED)
            {
                DNN_Query = new DNN(ParameterSetting.FEATURE_DIMENSION_QUERY,
                    ParameterSetting.SOURCE_LAYER_DIM,
                    ParameterSetting.SOURCE_ACTIVATION,
                    ParameterSetting.SOURCE_LAYERWEIGHT_SIGMA,
                    ParameterSetting.SOURCE_ARCH,
                    ParameterSetting.SOURCE_ARCH_WIND,
                    false);
                if (ParameterSetting.IS_SHAREMODEL)
                {
                    DNN_Doc = DNN_Query;
                }
                else
                {
                    DNN_Doc = new DNN(ParameterSetting.FEATURE_DIMENSION_DOC,
                        ParameterSetting.TARGET_LAYER_DIM,
                        ParameterSetting.TARGET_ACTIVATION,
                        ParameterSetting.TARGET_LAYERWEIGHT_SIGMA,
                        ParameterSetting.TARGET_ARCH,
                        ParameterSetting.TARGET_ARCH_WIND,
                        false);
                }

                DNN_Query.Init();

                if (!ParameterSetting.IS_SHAREMODEL)
                {
                    if (ParameterSetting.MIRROR_INIT)
                    {
                        DNN_Doc.Init(DNN_Query);
                    }
                    else
                    {
                        DNN_Doc.Init();
                    }
                }
                ParameterSetting.FEATURE_DIMENSION_QUERY = DNN_Query.neurallayers[0].Number;
                ParameterSetting.FEATURE_DIMENSION_DOC = DNN_Doc.neurallayers[0].Number;
            }
            else
            {
                LoadModel(ParameterSetting.SEEDMODEL1, ref DNN_Query, ParameterSetting.SEEDMODEL2, ref DNN_Doc, true);
            }

            Program.Print("Source Neural Network Structure " + DNN_Query.DNN_Descr());
            Program.Print("Target Neural Network Structure " + DNN_Doc.DNN_Descr());
            Program.Print("Feature Num Query " + ParameterSetting.FEATURE_DIMENSION_QUERY.ToString());
            Program.Print("Feature Num Doc " + ParameterSetting.FEATURE_DIMENSION_DOC.ToString());
            Program.Print("Sharing Model " + ParameterSetting.IS_SHAREMODEL.ToString());
            Program.Print("Mirror Init Model " + ParameterSetting.MIRROR_INIT.ToString());
            Program.Print("Math Lib " + ParameterSetting.MATH_LIB.ToString());
            if (ParameterSetting.MATH_LIB == MathLibType.cpu)
            {
                Program.Print("CPU Math thread num " + ParameterSetting.BasicMathLibThreadNum.ToString());
            }
        }

        void Init(DNN dnn_query, DNN dnn_doc)
        {
            dnn_model_query = new DNNRun(dnn_query);
            dnn_model_doc = new DNNRun(dnn_doc);

            alphaCudaPiece = new CudaPieceFloat(ParameterSetting.BATCH_SIZE * LabelDim, true, true);
            betaCudaPiece = new CudaPieceFloat(ParameterSetting.BATCH_SIZE * LabelDim, true, true);
        }

        unsafe public void Forward_CalClassification(BatchSample_Input query_batch, BatchSample_Input doc_batch)
        {
            /// forward (query doc, negdoc) streaming.
            dnn_model_query.forward_activate(query_batch);
            dnn_model_doc.forward_activate(doc_batch);
            //dnn_model_query.neurallayers.Last().Output.CopyOutFromCuda();
            //dnn_model_doc.neurallayers.Last().Output.CopyOutFromCuda();

            MathOperatorManager.GlobalInstance.Cosine_Similarity_SubSpace(dnn_model_query.neurallayers.Last().Output,
                                                dnn_model_doc.neurallayers.Last().Output,
                                                alphaCudaPiece, LabelDim, ParameterSetting.BATCH_SIZE, query_batch.batchsize,
                                                SubspaceDim, ParameterSetting.DSSMEpsilon);
            //alphaCudaPiece.CopyOutFromCuda();
            MathOperatorManager.GlobalInstance.SoftMax(alphaCudaPiece, betaCudaPiece, LabelDim, query_batch.batchsize, ParameterSetting.PARM_GAMMA);
            //MathOperatorManager.GlobalInstance.Cosine_Similarity(dnn_model_query.neurallayers.Last().Output,
            //        dnn_model_doc.neurallayers.Last().Output, alphaCudaPiece, ParameterSetting.NTRIAL + 1, ParameterSetting.BATCH_SIZE, 0,
            //        query_batch.batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon); // float.Epsilon);

        }

        public override float Evaluate()
        {
            PairValidStream.Init_Batch();
            PairValidStream.Eval_Init();
            while (PairValidStream.Next_Batch(SrcNorm, TgtNorm))
            {
                Forward_CalClassification(PairValidStream.qstream.Data, PairValidStream.dstream.Data);
                betaCudaPiece.CopyOutFromCuda();
                PairValidStream.Eval_Ouput_Batch(betaCudaPiece.MemPtr, null, new int[] { PairValidStream.qstream.Data.batchsize, LabelDim });
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


        unsafe public void calculate_deltaQD_TOP(int batchsize)
        {
            if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Tanh)
            {
                MathOperatorManager.GlobalInstance.Deriv_Cosine_Subspace(dnn_model_query.neurallayers.Last().Output,
                                              dnn_model_doc.neurallayers.Last().Output,
                                              dnn_model_query.neurallayers.Last().ErrorDeriv,
                                              dnn_model_doc.neurallayers.Last().ErrorDeriv,
                                              betaCudaPiece, 0, batchsize, LabelDim, SubspaceDim, -ParameterSetting.PARM_GAMMA, ParameterSetting.DSSMEpsilon);

                //MathOperatorManager.GlobalInstance.Deriv_Cosine(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
                //                    output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
                //                    batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
            }
            else if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Linear)
            {
                MathOperatorManager.GlobalInstance.Deriv_Cosine_Subspace(dnn_model_query.neurallayers.Last().Output,
                                              dnn_model_doc.neurallayers.Last().Output,
                                              dnn_model_query.neurallayers.Last().ErrorDeriv,
                                              dnn_model_doc.neurallayers.Last().ErrorDeriv,
                                              betaCudaPiece, 1, batchsize, LabelDim, SubspaceDim, -ParameterSetting.PARM_GAMMA, ParameterSetting.DSSMEpsilon);

                //MathOperatorManager.GlobalInstance.Derive_Cosine_Linear(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
                //                    output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
                //                    batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
            }
            else if (dnn_model_query.neurallinks.Last().NeuralLinkModel.Af == A_Func.Rectified)
            {
                MathOperatorManager.GlobalInstance.Deriv_Cosine_Subspace(dnn_model_query.neurallayers.Last().Output,
                                              dnn_model_doc.neurallayers.Last().Output,
                                              dnn_model_query.neurallayers.Last().ErrorDeriv,
                                              dnn_model_doc.neurallayers.Last().ErrorDeriv,
                                              betaCudaPiece, 2, batchsize, LabelDim, SubspaceDim, -ParameterSetting.PARM_GAMMA, ParameterSetting.DSSMEpsilon);

                //MathOperatorManager.GlobalInstance.Derive_Cosine_Rectified(dnn_model_query.neurallayers.Last().Output, dnn_model_doc.neurallayers.Last().Output,
                //                    output.cuda_layer_Deriv_Q, output.cuda_layer_Deriv_D,
                //                    batchsize, dnn_model_query.OutputLayerSize, ParameterSetting.DSSMEpsilon);
            }
        }

        unsafe public float feedstream_batch(BatchSample_Input query_batch, BatchSample_Input doc_batch, bool train_update)
        {
            /// forward (query doc, negdoc) streaming.
            Forward_CalClassification(query_batch, doc_batch);

            float error = 0;
            //if (ParameterSetting.LOSS_REPORT == 1)
            //{
            betaCudaPiece.CopyOutFromCuda();
            for (int i = 0; i < query_batch.batchsize; i++)
            {
                float mlambda = (float)Math.Log(betaCudaPiece.MemPtr[i * LabelDim + (int)Label[LabelCursor + i]] + ParameterSetting.DSSMEpsilon);
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
                betaCudaPiece.MemPtr[i * LabelDim + (int)Label[LabelCursor + i]] = betaCudaPiece.MemPtr[i * LabelDim + (int)Label[LabelCursor + i]] - 1;
                error += mlambda;
            }
            betaCudaPiece.CopyIntoCuda();
            //}
            if (train_update)
            {
                /******* Calculate the error derivatives on the top layer outputs *****/
                calculate_deltaQD_TOP(query_batch.batchsize);

                // back propagate 
                dnn_model_query.backward_propagate_deriv(query_batch);
                dnn_model_doc.backward_propagate_deriv(doc_batch);

                // update here 
                // here we have to do all the backprop computations before updating the model, because the model's actual weights will affect the backprop computation                
                dnn_model_query.update_weight(LearningParameters.momentum, LearningParameters.learning_rate * query_batch.batchsize / ParameterSetting.BATCH_SIZE);
                dnn_model_doc.update_weight(LearningParameters.momentum, LearningParameters.learning_rate * query_batch.batchsize / ParameterSetting.BATCH_SIZE);

                // and now it should support shared models
            }
            return error;
        }


        public override void Training()
        {
            Init(DNN_Query, DNN_Doc);
            DNN dnn_query_backup = null, dnn_doc_backup = null;
            Program.Print("Starting DNN Learning!");

            float trainingLoss = 0;

            float previous_devEval = 0;
            float VALIDATION_Eval = 0;
            //// determin the last stopped iteration

            Program.Print("Initialization (Iter 0)");
            Program.Print("Saving models ...");
            DNN_Query.CopyOutFromCuda();
            DNN_Query.Model_Save(ParameterSetting.MODEL_PATH + "_QUERY_ITER" + 0.ToString());
            if (!ParameterSetting.IS_SHAREMODEL)
            {
                DNN_Doc.CopyOutFromCuda();
                DNN_Doc.Model_Save(ParameterSetting.MODEL_PATH + "_DOC_ITER" + 0.ToString());
            }
            if (ParameterSetting.ISVALIDATE)
            {
                VALIDATION_Eval = Evaluate();
                Program.Print("Dataset VALIDATION :\n/*******************************/ \n" + VALIDATION_Eval.ToString() + " \n/*******************************/ \n");
            }
            File.WriteAllText(ParameterSetting.MODEL_PATH + "_LEARNING_RATE_ITER" + 0.ToString(), LearningParameters.lr_mid.ToString());

            //// Clone to backup models
            if (ParameterSetting.ISVALIDATE)
            {
                dnn_query_backup = (DNN)DNN_Query.CreateBackupClone();
                if (!ParameterSetting.IS_SHAREMODEL)
                {
                    dnn_doc_backup = (DNN)DNN_Doc.CreateBackupClone();
                }
            }

            if (ParameterSetting.NOTrain)
            {
                return;
            }
            Program.Print("total query sample number : " + PairStream.qstream.total_Batch_Size.ToString());
            Program.Print("total doc sample number : " + PairStream.dstream.total_Batch_Size.ToString());
            Program.Print("Training batches: " + PairStream.qstream.BATCH_NUM.ToString());
            Program.Print("Learning Objective : " + ParameterSetting.OBJECTIVE.ToString());
            LearningParameters.total_doc_num = PairStream.dstream.total_Batch_Size;

            previous_devEval = VALIDATION_Eval;

            Program.Print("Start Training");
            Program.Print("-----------------------------------------------------------");
            int mmindex = 0;
            for (int iter = 1; iter <= ParameterSetting.MAX_ITER; iter++)
            {
                Program.Print("ITER : " + iter.ToString());
                LearningParameters.learning_rate = LearningParameters.lr_mid;
                LearningParameters.momentum = 0.0f;

                Program.timer.Reset();
                Program.timer.Start();

                /// adjust learning rate here.
                PairStream.Init_Batch();
                trainingLoss = 0;
                mmindex = 0;
                LabelCursor = 0;
                while (PairStream.Next_Batch(SrcNorm, TgtNorm))
                {
                    trainingLoss += feedstream_batch(PairStream.GPU_qbatch, PairStream.GPU_dbatch, true);
                    LabelCursor += PairStream.GPU_qbatch.batchsize;
                    mmindex += 1;
                    if (mmindex % 50 == 0)
                    {
                        Console.Write("Training :{0}\r", mmindex.ToString());
                    }
                }

                Program.Print("Training Loss : " + trainingLoss.ToString());
                Program.Print("Learning Rate : " + (LearningParameters.learning_rate.ToString()));

                Program.Print("Saving models ...");
                DNN_Query.CopyOutFromCuda();
                DNN_Query.Model_Save(ParameterSetting.MODEL_PATH + "_QUERY_ITER" + iter.ToString());
                if (!ParameterSetting.IS_SHAREMODEL)
                {
                    DNN_Doc.CopyOutFromCuda();
                    DNN_Doc.Model_Save(ParameterSetting.MODEL_PATH + "_DOC_ITER" + iter.ToString());
                }

                if (ParameterSetting.ISVALIDATE)
                {
                    VALIDATION_Eval = Evaluate();
                    Program.Print("Dataset VALIDATION :\n/*******************************/ \n" + VALIDATION_Eval.ToString() + " \n/*******************************/ \n");

                    if (VALIDATION_Eval >= previous_devEval - LearningParameters.accept_range)
                    {
                        Program.Print("Accepted it");
                        previous_devEval = VALIDATION_Eval;
                        if (LearningParameters.IsrateDown)
                        {
                            LearningParameters.lr_mid = LearningParameters.lr_mid * LearningParameters.down_rate;
                        }
                        //// save model to backups
                        dnn_query_backup.Init(DNN_Query);
                        if (!ParameterSetting.IS_SHAREMODEL)
                        {
                            dnn_doc_backup.Init(DNN_Doc);
                        }
                    }
                    else
                    {
                        Program.Print("Reject it");

                        LearningParameters.IsrateDown = true;
                        LearningParameters.lr_mid = LearningParameters.lr_mid * LearningParameters.reject_rate;

                        //// recover model from the last saved backup
                        DNN_Query.Init(dnn_query_backup);
                        if (!ParameterSetting.IS_SHAREMODEL)
                        {
                            DNN_Doc.Init(dnn_doc_backup);
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
            DNN_Query.CopyOutFromCuda();
            DNN_Query.Model_Save(ParameterSetting.MODEL_PATH + "_QUERY_DONE");
            if (!ParameterSetting.IS_SHAREMODEL)
            {
                DNN_Doc.CopyOutFromCuda();
                DNN_Doc.Model_Save(ParameterSetting.MODEL_PATH + "_DOC_DONE");
            }
            //pstream.General_Train_Test(ParameterSetting.TRAIN_TEST_RATE);
            //dnn_train
        }

        public override void LoadValidateData(string[] files)
        {
            Program.timer.Reset();
            Program.timer.Start();

            PairValidStream.Load_Validate_PairData(files[0], files[1], files[2], Evaluation_Type.ClassficationScore);

            //ParameterSetting.VALIDATE_QFILE, ParameterSetting.VALIDATE_DFILE, ParameterSetting.VALIDATE_QDPAIR);
            Program.timer.Stop();
            Program.Print("loading Validate doc query stream done : " + Program.timer.Elapsed.ToString());
        }

        public override void LoadTrainData(string[] files)
        {
            Program.timer.Reset();
            Program.timer.Start();

            PairStream.Load_Train_PairData(files[0], files[1]);
            SrcNorm = Normalizer.CreateFeatureNormalize((NormalizerType)ParameterSetting.Q_FEA_NORM, PairStream.qstream.Feature_Size);
            TgtNorm = Normalizer.CreateFeatureNormalize((NormalizerType)ParameterSetting.D_FEA_NORM, PairStream.dstream.Feature_Size);
            PairStream.InitFeatureNorm(SrcNorm, TgtNorm);

            Label = new float[PairStream.qstream.total_Batch_Size];
            int Line_Idx = 0;
            StreamReader labelReader = new StreamReader(files[2]);
            while (!labelReader.EndOfStream)
            {
                int l = int.Parse(labelReader.ReadLine());
                Label[Line_Idx] = l;
                Line_Idx += 1;
                if (l >= LabelDim)
                    LabelDim = l + 1;
            }
            labelReader.Close();

            if (ParameterSetting.MIRROR_INIT)
            {
                int featureDim = Math.Max(ParameterSetting.FEATURE_DIMENSION_QUERY, ParameterSetting.FEATURE_DIMENSION_DOC);
                Program.Print(string.Format("Warning! MIRROR_INIT is turned on. Make sure two input sides are on the same feature space, and two models have exactly the same structure. Originally Feature Num Query {0}, Feature Num Doc {1}. Now both aligned to {2}", ParameterSetting.FEATURE_DIMENSION_QUERY, ParameterSetting.FEATURE_DIMENSION_DOC, featureDim));
                ParameterSetting.FEATURE_DIMENSION_QUERY = featureDim;
                ParameterSetting.FEATURE_DIMENSION_DOC = featureDim;
            }
            Program.timer.Stop();
            Program.Print("loading Training doc query stream done : " + Program.timer.Elapsed.ToString());
        }
    }
}
