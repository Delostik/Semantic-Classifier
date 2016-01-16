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
    public enum Evaluation_Type
    {
        PairScore,
        MultiRegressionScore,
        ClassficationScore,
        PairRegressioinScore
    }

    public class SequenceInputStream : IDisposable
    {
        public BatchSample_Input Data = null;

        public int nLine = 0;
        public int MAXSEGMENT_BATCH = 0;
        public int Feature_Size = ParameterSetting.FIXED_FEATURE_DIM;
        public int BATCH_NUM = 0;
        public int BATCH_INDEX = 0;
        public int LAST_INCOMPLETE_BATCH_SIZE = 0;

        FileStream mstream = null;
        BinaryReader mreader = null;

        ~SequenceInputStream()
        {
            Dispose();
        }

        public void CloseStream()
        {
            if (mstream != null)
            {
                mreader.Close();
                mstream.Close();
                mreader = null;
                mstream = null;
            }
            if (Data != null)
            {
                Data.Dispose();
                Data = null;
            }
        }

        public static int get_maxSegsize(string fileName)
        {
            FileStream mstreamt = new FileStream(fileName, FileMode.Open, FileAccess.Read);
            BinaryReader mreadert = new BinaryReader(mstreamt);
            mstreamt.Seek(-1 * sizeof(Int32), SeekOrigin.End);
            int max_Segsize = mreadert.ReadInt32();
            mreadert.Close();
            mstreamt.Close();
            mreadert = null;
            mstreamt = null;
            return max_Segsize;
        }

        public void get_dimension(string fileName)
        {
            mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
            mreader = new BinaryReader(mstream);
            mstream.Seek(-3 * sizeof(Int32), SeekOrigin.End);

            nLine = mreader.ReadInt32(); 
            int batch_size = mreader.ReadInt32();
  
            if (batch_size != ParameterSetting.BATCH_SIZE)
            {
                throw new Exception(string.Format(
                    "Batch_Size does not match between configuration and input data!\n\tFrom config: {0}.\n\tFrom data ({1}): {2}"
                    , ParameterSetting.BATCH_SIZE, fileName, batch_size)
                );
            }
            MAXSEGMENT_BATCH = mreader.ReadInt32();

            Data = new BatchSample_Input(ParameterSetting.BATCH_SIZE, MAXSEGMENT_BATCH);

            BATCH_NUM = (nLine + ParameterSetting.BATCH_SIZE - 1) / ParameterSetting.BATCH_SIZE;
            LAST_INCOMPLETE_BATCH_SIZE = nLine % ParameterSetting.BATCH_SIZE;
            BATCH_INDEX = 0;
        }

        public void Init()
        {
            BATCH_INDEX = 0;
            mstream.Seek(0, SeekOrigin.Begin);
        }

        void LoadDataBatch()
        {
            int expectedBatchSize = ParameterSetting.BATCH_SIZE;
            if (BATCH_INDEX == BATCH_NUM - 1 && LAST_INCOMPLETE_BATCH_SIZE != 0)
            {
                // only when the lastbatch is less than  BATCH_SIZE, we will need some care
                expectedBatchSize = LAST_INCOMPLETE_BATCH_SIZE;
            }
            Data.Load(mreader, expectedBatchSize);
        }

        public bool Fill()
        {
            if (BATCH_INDEX == BATCH_NUM)
            {
                return false;
            }
            LoadDataBatch();
            BATCH_INDEX++;
            return true;
        }

        public void Dispose()
        {
            CloseStream();
        }
    }

    public class LabeledSequenceInputStream : IDisposable
    {
        public LabeledBatchSample_Input Data = null;

        public int nLine = 0;
        public int MAXSEGMENT_BATCH = 0;
        public int Feature_Size = ParameterSetting.FIXED_FEATURE_DIM;
        public int BATCH_NUM = 0;
        public int BATCH_INDEX = 0;
        public int LAST_INCOMPLETE_BATCH_SIZE = 0;

        FileStream mstream = null;
        BinaryReader mreader = null;

        ~LabeledSequenceInputStream()
        {
            Dispose();
        }

        public void CloseStream()
        {
            if (mstream != null)
            {
                mreader.Close();
                mstream.Close();
                mreader = null;
                mstream = null;
            }
            if (Data != null)
            {
                Data.Dispose();
                Data = null;
            }
        }

        public void get_dimension(string fileName)
        {
            mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
            mreader = new BinaryReader(mstream);
            mstream.Seek(-3 * sizeof(Int32), SeekOrigin.End);

            nLine = mreader.ReadInt32();
            int batch_size = mreader.ReadInt32();

            if (batch_size != ParameterSetting.BATCH_SIZE)
            {
                throw new Exception(string.Format(
                    "Batch_Size does not match between configuration and input data!\n\tFrom config: {0}.\n\tFrom data ({1}): {2}"
                    , ParameterSetting.BATCH_SIZE, fileName, batch_size)
                );
            }
            MAXSEGMENT_BATCH = mreader.ReadInt32();

            Data = new LabeledBatchSample_Input(ParameterSetting.BATCH_SIZE, MAXSEGMENT_BATCH);

            BATCH_NUM = (nLine + ParameterSetting.BATCH_SIZE - 1) / ParameterSetting.BATCH_SIZE;
            LAST_INCOMPLETE_BATCH_SIZE = nLine % ParameterSetting.BATCH_SIZE;
            BATCH_INDEX = 0;
        }

        public int get_dimension2(string fileName)
        {
            mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
            mreader = new BinaryReader(mstream);
            mstream.Seek(-3 * sizeof(Int32), SeekOrigin.End);

            nLine = mreader.ReadInt32();
            int batch_size = mreader.ReadInt32();

            
            MAXSEGMENT_BATCH = mreader.ReadInt32();

            Data = new LabeledBatchSample_Input(batch_size, MAXSEGMENT_BATCH);

            BATCH_NUM = (nLine + ParameterSetting.BATCH_SIZE - 1) / ParameterSetting.BATCH_SIZE;
            LAST_INCOMPLETE_BATCH_SIZE = nLine % ParameterSetting.BATCH_SIZE;
            BATCH_INDEX = 0;
            return batch_size;
        }

        public void Init()
        {
            BATCH_INDEX = 0;
            mstream.Seek(0, SeekOrigin.Begin);
        }

        void LoadDataBatch()
        {
            int expectedBatchSize = ParameterSetting.BATCH_SIZE;
            if (BATCH_INDEX == BATCH_NUM - 1 && LAST_INCOMPLETE_BATCH_SIZE != 0)
            {
                // only when the lastbatch is less than  BATCH_SIZE, we will need some care
                expectedBatchSize = LAST_INCOMPLETE_BATCH_SIZE;
            }
            Data.Load(mreader, expectedBatchSize);
        }

        public bool Fill()
        {
            if (BATCH_INDEX == BATCH_NUM)
            {
                return false;
            }
            LoadDataBatch();
            BATCH_INDEX++;
            return true;
        }

        public void Dispose()
        {
            CloseStream();
        }
    }

    public class EvaluationSet
    {
        public static EvaluationSet Create(Evaluation_Type type)
        {
            EvaluationSet eval = null;
            switch (type)
            {
                case Evaluation_Type.PairScore:
                    eval = new PairScoreEvaluationSet();
                    break;
                case Evaluation_Type.MultiRegressionScore:
                    eval = new MultiRegressionEvaluationSet();
                    break;
                case Evaluation_Type.ClassficationScore:
                    eval = new ClassificationEvaluationSet();
                    break;
                case Evaluation_Type.PairRegressioinScore:
                    eval = new RegressionEvaluationSet();
                    break;
            }
            return eval;
        }

        public virtual void Loading_LabelInfo(string[] files)
        { }

        public virtual void Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        { }

        public virtual void Init()
        { }

        public virtual void Save(string scoreFile)
        { }

        static void CallExternalMetricEXE(string executiveFile, string arguments, string metricEvalResultFile)
        {
            if (File.Exists(metricEvalResultFile))
            {
                // remove previous result
                File.Delete(metricEvalResultFile);
            }
            using (Process callProcess = new Process()
            {
                StartInfo = new ProcessStartInfo()
                {
                    FileName = executiveFile,
                    Arguments = arguments,
                    CreateNoWindow = false,
                    UseShellExecute = true,
                }
            })
            {
                callProcess.Start();
                callProcess.WaitForExit();
            }
        }

        static float ReadExternalObjectiveMetric(string metricEvalResultFile, out List<string> validationFileLines)
        {
            // the first line should be a float, specifying the metric score
            if (!File.Exists(metricEvalResultFile))
            {
                throw new Exception(string.Format("Missing objective metric result file {0}, check your validation evaluation process!", metricEvalResultFile));
            }
            validationFileLines = new List<string>();

            StreamReader sr = new StreamReader(metricEvalResultFile);
            string line = sr.ReadLine();    // read the first line only
            float objectiveMetric = 0;
            if (!float.TryParse(line, out objectiveMetric))
                throw new Exception(string.Format("Cannot read objective metric from the result file {0}, check your validation evaluation process!", metricEvalResultFile));
            validationFileLines.Add(line);
            while ((line = sr.ReadLine()) != null)
            {
                validationFileLines.Add(line);
            }
            sr.Close();

            return objectiveMetric;
        }

        public float Evaluation(out List<string> validationFileLines)
        {
            string pairScoreFile = Path.GetRandomFileName();
            string objectiveMetricFile = pairScoreFile + ".metric";
            Program.Print("Saving validation prediction scores ... ");

            Save(pairScoreFile);
            //PairValidStream.SavePairPredictionScore(pairScoreFile);

            Program.Print("Calling external validation process ... ");
            CallExternalMetricEXE(ParameterSetting.VALIDATE_PROCESS, string.Format("{0} {1}", pairScoreFile, objectiveMetricFile), objectiveMetricFile);

            Program.Print("Reading validation objective metric  ... ");
            validationFileLines = null;
            float result = ReadExternalObjectiveMetric(objectiveMetricFile, out validationFileLines);

            if (File.Exists(pairScoreFile))
            {
                File.Delete(pairScoreFile);
            }

            if (File.Exists(objectiveMetricFile))
            {
                File.Delete(objectiveMetricFile);
            }

            return result;
        }

        public static float EvaluationModelOnly(string srcModelPath, string tgtModelPath, out List<string> validationFileLines)
        {
            string validationResultFile = Path.GetRandomFileName();

            Program.Print("Calling external validation process ... ");
            PairScoreEvaluationSet.CallExternalMetricEXE(ParameterSetting.VALIDATE_PROCESS, string.Format("{0} {1} {2}", srcModelPath, tgtModelPath, validationResultFile), validationResultFile);

            Program.Print("Reading validation objective metric  ... ");

            float result = PairScoreEvaluationSet.ReadExternalObjectiveMetric(validationResultFile, out validationFileLines);

            if (File.Exists(validationResultFile))
            {
                File.Delete(validationResultFile);
            }

            return result;
        }
    }

    public class PairScoreEvaluationSet : EvaluationSet
    {
        public List<string> PairInfo_Details = new List<string>();
        public string PairInfo_Header = string.Empty;
        public List<float> Pair_Score = new List<float>();

        public override void Loading_LabelInfo(string[] files)
        {
            FileStream mstream = new FileStream(files[0], FileMode.Open, FileAccess.Read);
            StreamReader mreader = new StreamReader(mstream);

            string mline = mreader.ReadLine();
            int Line_Idx = 0;
            if (!mline.Contains("m:"))
            {
                PairInfo_Details.Add(mline);
                Line_Idx += 1;
            }
            else
            {
                PairInfo_Header = mline;
            }
            while (!mreader.EndOfStream)
            {
                mline = mreader.ReadLine();
                PairInfo_Details.Add(mline);
                Line_Idx += 1;
            }
            mreader.Close();
            mstream.Close();
        }

        public override void Save(string scoreFile)
        {
            StreamWriter mtwriter = new StreamWriter(scoreFile);
            mtwriter.WriteLine(PairInfo_Header + "\tDSSM_Score"); // header
            for (int i = 0; i < Pair_Score.Count; i++)
            {
                float v = Pair_Score[i];
                mtwriter.WriteLine(PairInfo_Details[i] + "\t" + v.ToString());
            }
            mtwriter.Close();
        }

        public override void Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            for (int i = 0; i < args[0]; i++)
            {
                Pair_Score.Add(score[i]);
            }
        }

        public override void Init()
        {
            Pair_Score.Clear();
        }
    }

    public class MultiRegressionEvaluationSet : EvaluationSet
    {
        public List<string> PairInfo_Details = new List<string>();
        public List<float> Pair_Score = new List<float>();
        public int Dimension = 0;
        public override void Init()
        {
            PairInfo_Details.Clear();
            Pair_Score.Clear();
        }

        public override void Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            Dimension = args[1];
            for (int i = 0; i < args[0]; i++)
            {
                StringBuilder sb = new StringBuilder();
                for (int k = 0; k < args[1]; k++)
                {
                    Pair_Score.Add(score[i * args[1] + k]);
                    sb.Append(groundTrue[i * args[1] + k].ToString() + ",");
                }
                PairInfo_Details.Add(sb.ToString());
            }
        }

        public override void Save(string scoreFile)
        {
            StreamWriter mtwriter = new StreamWriter(scoreFile);
            for (int i = 0; i < Pair_Score.Count / Dimension; i++)
            {
                StringBuilder sb = new StringBuilder();
                for (int k = 0; k < Dimension; k++)
                {
                    sb.Append(Pair_Score[i * Dimension + k].ToString() + ",");
                }
                mtwriter.WriteLine(PairInfo_Details[i].ToString() + "\t" + sb.ToString());
            }
            mtwriter.Close();
        }
    }

    public class RegressionEvaluationSet : EvaluationSet
    {
        public List<float> PairInfo_Details = new List<float>();
        public List<float> Pair_Scores = new List<float>();

        public override void Loading_LabelInfo(string[] files)
        {
            FileStream mstream = new FileStream(files[0], FileMode.Open, FileAccess.Read);
            StreamReader mreader = new StreamReader(mstream);

            int Line_Idx = 0;
            while (!mreader.EndOfStream)
            {
                string mline = mreader.ReadLine();
                float label = float.Parse(mline);
                PairInfo_Details.Add(label);
                Line_Idx += 1;
            }
            mreader.Close();
            mstream.Close();
        }

        public override void Init()
        {
            Pair_Scores.Clear();
        }

        public override void Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            for (int i = 0; i < args[0]; i++)
            {
                float f = score[i];
                Pair_Scores.Add(f);
            }
        }

        public override void Save(string scoreFile)
        {
            StreamWriter mtwriter = new StreamWriter(scoreFile);
            for (int i = 0; i < PairInfo_Details.Count; i++)
            {
                float g = PairInfo_Details[i];
                float p = Pair_Scores[i];
                mtwriter.WriteLine(g.ToString() + "\t" + p.ToString());
            }
            mtwriter.Close();
        }

    }

    public class ClassificationEvaluationSet : EvaluationSet
    {
        public List<int> PairInfo_Details = new List<int>();
        public List<float[]> Pair_Scores = new List<float[]>();
        public override void Loading_LabelInfo(string[] files)
        {
            FileStream mstream = new FileStream(files[0], FileMode.Open, FileAccess.Read);
            StreamReader mreader = new StreamReader(mstream);

            int Line_Idx = 0;
            while (!mreader.EndOfStream)
            {
                string mline = mreader.ReadLine();
                int label = int.Parse(mline);
                PairInfo_Details.Add(label);
                Line_Idx += 1;
            }
            mreader.Close();
            mstream.Close();
        }

        public override void Save(string scoreFile)
        {
            StreamWriter mtwriter = new StreamWriter(scoreFile);
            for (int i = 0; i < PairInfo_Details.Count; i++)
            {
                StringBuilder sb = new StringBuilder();
                float v = PairInfo_Details[i];

                sb.Append(v.ToString() + "\t");
                for (int k = 0; k < Pair_Scores[i].Length; k++)
                {
                    sb.Append(Pair_Scores[i][k].ToString() + "\t");
                }

                mtwriter.WriteLine(sb.ToString());
            }
            mtwriter.Close();
        }

        public override void Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            for (int i = 0; i < args[0]; i++)
            {
                StringBuilder sb = new StringBuilder();
                float[] f = new float[args[1]];

                for (int k = 0; k < args[1]; k++)
                {
                    f[k] = score[i * args[1] + k];
                }
                Pair_Scores.Add(f);
            }
        }

        public override void Init()
        {
            Pair_Scores.Clear();
        }
    }



    public class PairInputStream : IDisposable
    {
        public SequenceInputStream q0stream = new SequenceInputStream();
        public SequenceInputStream q1stream = new SequenceInputStream();
        public SequenceInputStream q2stream = new SequenceInputStream();

        public static int MAXSEGMENT_BATCH = 40000;
        //public static int QUERY_MAXSEGMENT_BATCH = 40000;
        //public static int DOC_MAXSEGMENT_BATCH = 40000;

        /********How to transform the qbatch, dbatch and negdbatch into GPU Memory**********/
        public BatchSample_Input GPU_q0batch { get { return q0stream.Data; } }
        public BatchSample_Input GPU_q1batch { get { return q1stream.Data; } }
        public BatchSample_Input GPU_q2batch { get { return q2stream.Data; } }
        /*************************************************************************************/

        /**************** Associated streams *************/
        public StreamReader srNCEProbDist = null;

        ~PairInputStream()
        {
            Dispose();
        }

        #region For Validation stuff

        EvaluationSet eval = null;
        public void Eval_Init()
        {
            eval.Init();
        }

        public void Eval_Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            eval.Ouput_Batch(score, groundTrue, args);
        }

        public float Eval_Score(out List<string> validationFileLines)
        {
            return eval.Evaluation(out validationFileLines);
        }

        public float Eval_Score_ModelOnlyEvaluationModelOnly(string srcModelPath, string tgtModelPath, out List<string> validationFileLines)
        {
            return EvaluationSet.EvaluationModelOnly(srcModelPath, tgtModelPath, out validationFileLines);
        }

        #endregion

        /// <summary>
        /// Used by valid input
        /// </summary>
        /// <param name="qFileName"></param>
        /// <param name="dFileName"></param>
        /// <param name="pairFileName"></param>
        public void Load_Validate_PairData(List<string> qFileName, string pairFileName, Evaluation_Type type)
        {
            Load_PairData(qFileName, null);

            eval = EvaluationSet.Create(type);

            eval.Loading_LabelInfo(new string[] { pairFileName });
        }
        /// <summary>
        /// Used by training input
        /// </summary>
        /// <param name="qFileName"></param>
        /// <param name="dFileName"></param>
        /// <param name="nceProbDistFile"></param>
        public void Load_Train_TriData(List<string> qFileName, string nceProbDistFile = null)
        {
            Load_PairData(qFileName, nceProbDistFile);

            //// We only update feature dimension from train stream on the first fresh kickoff
            //// whenever the feature dimensions have been set or load from models, we will skip the update here
            if (ParameterSetting.FEATURE_DEMENSION_Q0 <= 0 || ParameterSetting.FEATURE_DEMENSION_Q1 <= 0 || ParameterSetting.FEATURE_DEMENSION_Q2 <= 0)
            {
                ParameterSetting.FEATURE_DEMENSION_Q0 = q0stream.Feature_Size;
                ParameterSetting.FEATURE_DEMENSION_Q1 = q1stream.Feature_Size;
                ParameterSetting.FEATURE_DEMENSION_Q2 = q2stream.Feature_Size;

                //if (ParameterSetting.MIRROR_INIT)
                //{
                //    int featureDim = Math.Max(ParameterSetting.FEATURE_DEMENSION_Q0, ParameterSetting.FEATURE_DEMENSION_Q1);
                //    featureDim = Math.Max(featureDim, ParameterSetting.FEATURE_DEMENSION_Q2);

                //    Program.Print(string.Format("Warning! MIRROR_INIT is turned on. Make sure two input sides are on the same feature space, and two models have exactly the same structure. Originally Feature Num Query {0}, Feature Num Doc {1}. Now both aligned to {2}",
                //        ParameterSetting.FEATURE_DEMENSION_Q0, 
                //        ParameterSetting.FEATURE_DEMENSION_Q1, 
                //        ParameterSetting.FEATURE_DEMENSION_Q2, 
                //        featureDim));
                //    ParameterSetting.FEATURE_DEMENSION_Q0 = featureDim;
                //    ParameterSetting.FEATURE_DEMENSION_Q1 = featureDim;
                //    ParameterSetting.FEATURE_DEMENSION_Q2 = featureDim;
                //}
            }
        }

        void Load_PairData(List<string> qFileName, string nceProbDistFile)
        {
            CloseAllStreams();
            q0stream.get_dimension(qFileName[0]);
            q1stream.get_dimension(qFileName[1]);
            q2stream.get_dimension(qFileName[2]);
            if (nceProbDistFile != null)
            {
                this.srNCEProbDist = new StreamReader(nceProbDistFile);
            }

            MAXSEGMENT_BATCH = Math.Max(q0stream.MAXSEGMENT_BATCH, q1stream.MAXSEGMENT_BATCH);
            MAXSEGMENT_BATCH = Math.Max(q2stream.MAXSEGMENT_BATCH, MAXSEGMENT_BATCH);
        }

        public void Init_Batch()
        {
            q0stream.Init();
            q1stream.Init();
            q2stream.Init();
        }

        public bool Next_Batch()
        {
            if (!q0stream.Fill() || !q1stream.Fill() || !q2stream.Fill())
            {
                return false;
            }
            
            q0stream.Data.Batch_In_GPU();
            q1stream.Data.Batch_In_GPU();
            q2stream.Data.Batch_In_GPU();
            return true;
        }

        public void CloseAllStreams()
        {
            q0stream.CloseStream();
            q1stream.CloseStream();
            q2stream.CloseStream();

            if (this.srNCEProbDist != null)
            {
                this.srNCEProbDist.Close();
                this.srNCEProbDist = null;
            }
        }

        public void Dispose()
        {
            q0stream.Dispose();
            q1stream.Dispose();
            q2stream.Dispose();
            CloseAllStreams();
        }
    }

    public class LabeledInputStream : IDisposable
    {
        public LabeledSequenceInputStream lstream = new LabeledSequenceInputStream();

        public int BatchSize = 0;
        public bool isTrain;
        public int MAXSEGMENT_BATCH = 40000;
        //public static int QUERY_MAXSEGMENT_BATCH = 40000;
        //public static int DOC_MAXSEGMENT_BATCH = 40000;

        /********How to transform the qbatch, dbatch and negdbatch into GPU Memory**********/
        public LabeledBatchSample_Input GPU_lbatch { get { return lstream.Data; } }
        /*************************************************************************************/

        /**************** Associated streams *************/
        public StreamReader srNCEProbDist = null;

        public LabeledInputStream(bool isTrain)
        {
            this.isTrain = isTrain;
        }

        ~LabeledInputStream()
        {
            Dispose();
        }

        #region For Validation stuff

        EvaluationSet eval = null;
        public void Eval_Init()
        {
            eval.Init();
        }

        public void Eval_Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            eval.Ouput_Batch(score, groundTrue, args);
        }

        public float Eval_Score(out List<string> validationFileLines)
        {
            return eval.Evaluation(out validationFileLines);
        }

        public float Eval_Score_ModelOnlyEvaluationModelOnly(string srcModelPath, string tgtModelPath, out List<string> validationFileLines)
        {
            return EvaluationSet.EvaluationModelOnly(srcModelPath, tgtModelPath, out validationFileLines);
        }

        #endregion

        /// <summary>
        /// Used by valid input, not used~~~~~~~~~~~~~
        /// </summary>
        /// <param name="qFileName"></param>
        /// <param name="dFileName"></param>
        /// <param name="pairFileName"></param>
        public void Load_Validate_PairData(string lFileName, string pairFileName, Evaluation_Type type)
        {
            Load_PairData(lFileName, null);

            eval = EvaluationSet.Create(type);

            eval.Loading_LabelInfo(new string[] { pairFileName });
        }
        /// <summary>
        /// Used by training input
        /// </summary>
        /// <param name="qFileName"></param>
        /// <param name="dFileName"></param>
        /// <param name="nceProbDistFile"></param>
        public void Load_Train_TriData(string lFileName, string nceProbDistFile = null)
        {
            Load_PairData(lFileName, nceProbDistFile);

            //// We only update feature dimension from train stream on the first fresh kickoff
            //// whenever the feature dimensions have been set or load from models, we will skip the update here
            if (ParameterSetting.FEATURE_DEMENSION_L <= 0)
            {
                ParameterSetting.FEATURE_DEMENSION_L = lstream.Feature_Size;
            }
        }

        void Load_PairData(string lFileName, string nceProbDistFile)
        {
            CloseAllStreams();
            if (isTrain)
                lstream.get_dimension(lFileName);
            else
                BatchSize = lstream.get_dimension2(lFileName);
            if (nceProbDistFile != null)
            {
                this.srNCEProbDist = new StreamReader(nceProbDistFile);
            }

            MAXSEGMENT_BATCH = lstream.MAXSEGMENT_BATCH;
        }

        public void Init_Batch()
        {
            lstream.Init();
        }

        public bool Next_Batch()
        {
            if (!lstream.Fill())
            {
                return false;
            }
            lstream.Data.Batch_In_GPU();
            return true;
        }

        public void CloseAllStreams()
        {
            lstream.CloseStream();

            if (this.srNCEProbDist != null)
            {
                this.srNCEProbDist.Close();
                this.srNCEProbDist = null;
            }
        }

        public void Dispose()
        {
            lstream.Dispose();
            CloseAllStreams();
        }
    }
}
