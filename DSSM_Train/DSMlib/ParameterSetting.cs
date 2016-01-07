﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DSMlib
{
    public enum MathLibType
    {
        gpu,
        cpu
    }
    public enum ObjectiveType
    {
        WEAKRANK,
        SOFTMAX
    }

    public class ParameterSetting
    {

        // 下面这段是我加的
        public static int FIXED_FEATURE_DIM = 15;  // 输入词向量的长度

        public static bool CuBlasEnable = true;
        public static bool CheckGrad = false;

        public static ObjectiveType OBJECTIVE = ObjectiveType.WEAKRANK; //0:weak supervison training; 1:classic supervised training
        //public static string NCE_PROB_FILE = "_null_"; //if NCE and probFile="_null_" then use uniform Prob(D), e.g., Prob(D) = 1/|D|
        public static int LOSS_REPORT = 1; //report loss 
        public static string reserved_settings = ""; //no use yet
        
        public static int BATCH_SIZE = 1024;
        //public static int NTRIAL = 4;
        //public static float PARM_GAMMA = 10;
        public static float PARM_MARGIN = 1;
        public static int MAX_ITER = 40;

        public static int[] LAYER_DIM = { 300, 300, 128 };
        public static float[] LAYERWEIGHT_SIGMA = { 0.2f, 0.6f, 0.6f };
        public static int[] ACTIVATION = { 1, 1, 1 };
        public static int[] ARCH = { 0, 0, 0 };
        public static int[] ARCH_WND = { 1, 1, 1 };

        public static int[] ARCH_WNDS = { 1, 2, 3 };
        public static int[] ARCH_FMS = { 200, 200, 200 };

        public static int CONTEXT_DIM = 50;
        public static int CONTEXT_NUM = 50;
        public static int WORD_NUM = 200000;
        public static float BIAS_WEIGHT = 0;

        //public static int FEATURE_DIMENSION_QUERY = 0;
        //public static int FEATURE_DIMENSION_DOC = 0;
        public static int FEATURE_DEMENSION_Q0 = 0;
        public static int FEATURE_DEMENSION_Q1 = 0;
        public static int FEATURE_DEMENSION_Q2 = 0;

        public static int FEATURE_DEMENSION_L = 0;

        public static string SHALLOW_SOURCE = "";
        public static string SHALLOW_TARGET = "";
        public static bool IS_SHALLOW = false;
        public static bool IS_SHAREMODEL = false;

        //public static string QFILE = "";
        //public static int Q_FEA_NORM = 0;

        //public static string DFILE = "";
        //public static int D_FEA_NORM = 0;

        public static string QFILE_0 = "";
        public static int Q0_FEA_NORM = 0;
        public static string QFILE_1 = "";
        public static int Q1_FEA_NORM = 0;
        public static string QFILE_2 = "";
        public static int Q2_FEA_NORM = 0;

        public static string LFILE = "";

        public static bool ISVALIDATE = false;
        public static string VALIDATE_FILE = "";
        public static string VALIDATE_PROCESS = "";
        /// <summary>
        /// If true, then VALIDATE_PROCESS only need MODEL to valid. Don't need VALIDATE_QFILE, VALIDATE_DFILE, and VALIDATE_QDPAIR anymore
        /// </summary>
        public static bool VALIDATE_MODEL_ONLY = false;


        public static bool ISSEED = false;
        public static string SEEDMODEL1 = "";
        public static string SEEDMODEL2 = "";
        public static string SEEDMODEL3 = "";
        public static string SEEDMODEL4 = "";
        public static bool NOTrain = false;

        public static int device = 0;

        public static MathLibType MATH_LIB = MathLibType.gpu;
        public static int BasicMathLibThreadNum = 128;
        /// <summary>
        /// set to be true to be able to load the old model format generated by the original DSSM training.
        /// note that we always save model in the new format regardless of this parameter.
        /// by default it is false, meaning loading model in the new format.
        /// </summary>
        public static bool LoadModelOldFormat = false;

        /// <summary>
        /// For backward-compatibility of input data format.
        /// Possible values are "BOW", "SEQ", or "".
        /// "BOW" is used by 
        /// </summary>
        public static string LoadInputBackwardCompatibleMode = string.Empty;

        public static float DSSMEpsilon = 0.00000001f;

        public static bool Denoising = false;

        /// <summary>
        /// set to be true to load feature values as int32
        /// by default it is false, meaning loading feature value as single float. 
        /// </summary>
        //public static bool FeatureValueAsInt = false;

        //public static int Linear_Mapping = 0;

        /// <summary>
        /// 
        /// </summary>
        public static bool UpdateBias = false;

        public static string WORDLT_INIT = null;

        public static void LoadArgs(string conf_filename)
        {
            if (!File.Exists(conf_filename))
                return;
            FileStream mstream = new FileStream(conf_filename, FileMode.Open, FileAccess.Read);
            StreamReader mreader = new StreamReader(mstream);
            while (!mreader.EndOfStream)
            {
                string[] cmds = mreader.ReadLine().Split('\t');
                if (cmds.Length < 2)
                {
                    continue;
                }

                if (cmds[0].Equals("OBJECTIVE"))
                {
                    if (cmds[1].Trim().ToUpper() == "WEAKRANK")
                    {
                        OBJECTIVE = ObjectiveType.WEAKRANK;
                    }
                    else if (cmds[1].Trim().ToUpper() == "SOFTMAX")
                    {
                        OBJECTIVE = ObjectiveType.SOFTMAX;
                    }
                    else throw new Exception("NOT supported trainning objective!");
                }
                //else if (cmds[0].Equals("LINEAR_MAPPING"))
                //{
                //    Linear_Mapping = int.Parse(cmds[1].Trim().ToUpper());
                //}
                else if (cmds[0].Equals("LOSS_REPORT"))
                {
                    LOSS_REPORT = int.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("reserved_settings"))
                {
                    reserved_settings = cmds[1];
                }
                else if (cmds[0].Equals("CUBLAS"))
                {
                    if (int.Parse(cmds[1]) == 1)
                    {
                        CuBlasEnable = true;
                    }
                    else
                    {
                        CuBlasEnable = false;
                    }
                }
                else if (cmds[0].Equals("CHECK_GRADIENT"))
                {
                    if (int.Parse(cmds[1]) == 1)
                    {
                        CheckGrad = true;
                    }
                    else
                    {
                        CheckGrad = false;
                    }
                }
                else if (cmds[0].Equals("BATCHSIZE"))
                {
                    BATCH_SIZE = int.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("PARM_MARGIN"))
                {
                    PARM_MARGIN = float.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("BIDSSM"))
                {
                    if (int.Parse(cmds[1]) == 1)
                    {
                        IS_SHAREMODEL = false;
                    }
                    else
                    {
                        IS_SHAREMODEL = true;
                    }
                }                
                else if (cmds[0].Equals("MAX_ITER"))
                {
                    MAX_ITER = int.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("SHALLOW_SOURCE"))
                {
                    SHALLOW_SOURCE = cmds[1];
                    IS_SHALLOW = true;
                }
                else if (cmds[0].Equals("SHALLOW_TARGET"))
                {
                    SHALLOW_TARGET = cmds[1];
                    IS_SHALLOW = true;
                }
                else if (cmds[0].Equals("DEVICE"))
                {
                    device = int.Parse(cmds[1]);
                    Cudalib.CudaSetDevice(device);
                }
                else if (cmds[0].Equals("LFILE"))
                {
                    LFILE = cmds[1];
                }
                else if (cmds[0].Equals("WORDLT_INIT"))
                {
                    WORDLT_INIT = cmds[1];
                }
                else if (cmds[0].Equals("Q0FILE"))
                {
                    QFILE_0 = cmds[1];
                }
                else if (cmds[0].Equals("Q1FILE"))
                {
                    QFILE_1 = cmds[1];
                }
                else if (cmds[0].Equals("Q2FILE"))
                {
                    QFILE_2 = cmds[1];
                }
                else if (cmds[0].Equals("Q1_FEA_NORM"))
                {
                    Q1_FEA_NORM = int.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("Q2_FEA_NORm"))
                {
                    Q2_FEA_NORM = int.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("Q0_FEA_NORm"))
                {
                    Q0_FEA_NORM = int.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("CONTEXT_DIM"))
                {
                    CONTEXT_DIM = int.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("CONTEXT_NUM"))
                {
                    CONTEXT_NUM = int.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("WORD_NUM"))
                {
                    WORD_NUM = int.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("LOGFILE"))
                {
                    Log_FileName = cmds[1];
                }
                else if (cmds[0].Equals("LEARNINGRATE"))
                {
                    LearningParameters.lr_begin = float.Parse(cmds[1]);
                    LearningParameters.lr_mid = float.Parse(cmds[1]);
                    LearningParameters.lr_latter = float.Parse(cmds[1]);
                    LearningParameters.learning_rate = float.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("SEEDMODEL1"))
                {
                    SEEDMODEL1 = cmds[1];
                    ISSEED = true;
                    //NOTrain = true;
                }
                else if (cmds[0].Equals("SEEDMODEL2"))
                {
                    SEEDMODEL2 = cmds[1];
                    ISSEED = true;
                    //NOTrain = true;
                }
                else if (cmds[0].Equals("SEEDMODEL3"))
                {
                    SEEDMODEL3 = cmds[1];
                    ISSEED = true;
                }
                else if (cmds[0].Equals("SEEDMODEL4"))
                {
                    SEEDMODEL4 = cmds[1];
                    ISSEED = true;
                }
                else if (cmds[0].Equals("ARCH"))
                {
                    string[] items = cmds[1].Split(',');
                    ARCH = new int[items.Length];
                    int i = 0;
                    foreach (string s in items)
                    {
                        ARCH[i] = int.Parse(s);
                        i++;
                    }
                }
                else if (cmds[0].Equals("ARCH_WNDSIZE"))
                {
                    string[] items = cmds[1].Split(',');
                    ARCH_WND = new int[items.Length];
                    int i = 0;
                    foreach (string s in items)
                    {
                        ARCH_WND[i] = int.Parse(s);
                        i++;
                    }
                }
                else if (cmds[0].Equals("ARCH_FMSIZES"))
                {
                    string[] items = cmds[1].Split(',');
                    ARCH_FMS = new int[items.Length];
                    int i = 0;
                    foreach (string s in items)
                    {
                        ARCH_FMS[i] = int.Parse(s);
                        i++;
                    }
                }
                else if (cmds[0].Equals("ARCH_WNDSIZES"))
                {
                    string[] items = cmds[1].Split(',');
                    ARCH_WNDS = new int[items.Length];
                    int i = 0;
                    foreach (string s in items)
                    {
                        ARCH_WNDS[i] = int.Parse(s);
                        i++;
                    }
                }
                else if (cmds[0].Equals("LAYER_DIM"))
                {
                    string[] items = cmds[1].Split(',');
                    LAYER_DIM = new int[items.Length];
                    int i = 0;
                    foreach (string s in items)
                    {
                        LAYER_DIM[i] = int.Parse(s);
                        i++;
                    }
                }
                else if (cmds[0].Equals("LAYERWEIGHT_SIGMA"))
                {
                    string[] items = cmds[1].Split(',');
                    LAYERWEIGHT_SIGMA = new float[items.Length];
                    int i = 0;
                    foreach (string s in items)
                    {
                        LAYERWEIGHT_SIGMA[i] = float.Parse(s);
                        i++;
                    }
                }
                else if (cmds[0].Equals("ACTIVATION"))
                {
                    string[] items = cmds[1].Split(',');
                    ACTIVATION = new int[items.Length];
                    int i = 0;
                    foreach (string s in items)
                    {
                        ACTIVATION[i] = int.Parse(s);
                        i++;
                    }
                }                
                else if (cmds[0].Equals("VALIDATE_FILE"))
                {
                    VALIDATE_FILE = cmds[1];
                }
                else if (cmds[0].Equals("VALIDATEPROCESS"))
                {
                    VALIDATE_PROCESS = cmds[1];
                    ISVALIDATE = true;
                }
                else if (cmds[0].Equals("VALIDATE_MODEL_ONLY"))
                {
                    if (int.Parse(cmds[1]) == 1)
                    {
                        VALIDATE_MODEL_ONLY = true;
                    }
                    else
                    {
                        VALIDATE_MODEL_ONLY = false;
                    }
                }
                else if (cmds[0].Equals("MODELPATH"))
                {
                    MODEL_PATH = cmds[1];
                }
                else if (cmds[0].Equals("EVULATIONEXE"))
                {
                    EVULATION_EXE = cmds[1];
                }
                else if (cmds[0].Equals("LOAD_MODEL_OLD_FORMAT"))
                {
                    if (!bool.TryParse(cmds[1], out LoadModelOldFormat))
                    {
                        LoadModelOldFormat = cmds[1].Trim().Equals("1");
                    }
                }
                else if (cmds[0].Equals("REJECT_RATE"))
                {
                    LearningParameters.reject_rate = float.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("DOWN_RATE"))
                {
                    LearningParameters.down_rate = float.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("ACCEPT_RANGE"))
                {
                    LearningParameters.accept_range = float.Parse(cmds[1]);
                }
                else if (cmds[0].Equals("MATH_LIB"))
                {
                    if (cmds[1].Trim().ToUpper() == "CPU")
                    {
                        MATH_LIB = MathLibType.cpu;
                    }
                    else
                    {
                        MATH_LIB = MathLibType.gpu;
                    }
                }
                else if (cmds[0].Equals("CPU_MATH_LIB_THREAD_NUM"))
                {
                    ParameterSetting.BasicMathLibThreadNum = int.Parse(cmds[1]);
                    if (ParameterSetting.BasicMathLibThreadNum < 1)
                    {
                        throw new Exception("Error! CPU_MATH_LIB_THREAD_NUM should be >= 1");
                    }
                }
                else if (cmds[0].Equals("RANDOM_SEED"))
                {
                    RANDOM_SEED = int.Parse(cmds[1]);
                    if (RANDOM_SEED >= 0) PSEUDO_RANDOM = true;
                    else                  PSEUDO_RANDOM = false;
                }
                else if (cmds[0].Equals("UPDATE_BIAS"))
                {
                    if (int.Parse(cmds[1]) == 1)
                    {
                        UpdateBias = true;
                    }
                    else
                    {
                        UpdateBias = false;
                    }
                }
            }
            if (PSEUDO_RANDOM)
            {
                Random = new Random(RANDOM_SEED);
            }
            else
            {
                Random = new Random();
            }
            mreader.Close();
            mstream.Close();
        }

        public static string EVULATION_EXE = @"D:\t-yeshen\Dataset\SimpleNNEvaluate.exe";
        public static string MODEL_PATH = @"D:\t-yeshen\DSMLIB_Model_0616\DNN_MODEL";
        public static string Log_FileName = @"D:\t-yeshen\DSMLIB_Model_0616\LOG";

        public static bool PSEUDO_RANDOM = true;
        //public static int[] RANDOM_SEED = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        public static int RANDOM_SEED = 13;
        public static Random Random = null;
        
        public static bool DEBUG = false;
        public static int DEBUG_TRAIN_NUM = 20480000;
        public static int DEBUG_BATCH_NUM = 100;
    }
}
