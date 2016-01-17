using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.IO;
namespace DSMlib
{

    public class LearningParameters
    {
        public static float learning_rate = 0.1f;
        public static float lr_begin = 0.4f;
        public static float lr_mid = 0.4f;
        public static float lr_latter = 0.02f;
        public static float momentum = 0;
        public static float finalMomentum = 0.02f;
        public static float lrchange = 0.5f;
        public static bool IsrateDown = false;
        public static bool neg_static_sample = false;
        public static float reject_rate = 0.2f;
        public static float down_rate = 1.0f;
        public static float accept_range = 0.002f;
        public static int total_doc_num = 0;
        public static int learn_style = 0; // 0 : mini-batch; 1 : whole-batch;
    }

    class Program
    {
        [DllImport("msvcrt.dll")]
        static extern bool system(string str);

        static FileStream log_stream = null;//new FileStream(ParameterSetting.Log_FileName + "SEED" + ParameterSetting.RANDOMSEED.ToString(), FileMode.Create, FileAccess.Write);
        static StreamWriter log_writer = null; // new StreamWriter(log_stream);
        
        public static void Print(string mstr)
        {
            Console.WriteLine(mstr);
            if (log_writer != null)
            {
                log_writer.WriteLine(mstr);
                log_writer.Flush();
            }
        }

        public static Stopwatch timer = new Stopwatch();

        static void Main(string[] args)
        {
            try
            {
                ParameterSetting.LoadArgs("config.txt");

                if (args != null && args.Length > 0)
                    ParameterSetting.LoadArgs(args[0]);
                string logDirecotry = new FileInfo(ParameterSetting.Log_FileName).Directory.FullName;
                if (!Directory.Exists(logDirecotry))
                {
                    Directory.CreateDirectory(logDirecotry);
                }
                log_stream = new FileStream(ParameterSetting.Log_FileName, FileMode.Append, FileAccess.Write);
                log_writer = new StreamWriter(log_stream);

                string modelDirectory = new FileInfo(ParameterSetting.MODEL_PATH).Directory.FullName;
                if (!Directory.Exists(modelDirectory))
                {
                    Directory.CreateDirectory(modelDirectory);
                }

                //timer.Reset();
                //timer.Start();

                if (ParameterSetting.CuBlasEnable)
                {
                    Cudalib.CUBLAS_Init();
                }
                //Load_Train_PairData(ParameterSetting.QFILE, ParameterSetting.DFILE);

                if (ParameterSetting.CheckGrad)
                {
                    CheckGradient cg = new CheckGradient();
                    cg.initDNN();
                    cg.initRun();
                    cg.CheckGrad();
                    if (ParameterSetting.CuBlasEnable)
                        Cudalib.CUBLAS_Destroy();
                    return;
                }

                DNN_Train dnnTrain = null;

                Print("Loading training data ....");
               /// 1. loading training dataset.
                switch (ParameterSetting.OBJECTIVE)
                {                 
                    case ObjectiveType.WEAKRANK:
                        dnnTrain = new Weak_DNNTrain();
                        string[] trainfiles = new string[3] { ParameterSetting.QFILE_0, ParameterSetting.QFILE_1, ParameterSetting.QFILE_2 };
                        dnnTrain.LoadTrainData(trainfiles);
                        if (ParameterSetting.ISVALIDATE)
                        {
                            dnnTrain.LoadValidateData(new string[1] { ParameterSetting.VALIDATE_FILE });
                        }
                        break;
                    case ObjectiveType.SOFTMAX:
                        dnnTrain = new Label_DNNTrain();
                        string trainfile = ParameterSetting.QFILE_0;
                        string validfile = ParameterSetting.QFILE_1;
                        string testfile = ParameterSetting.QFILE_2;
                        dnnTrain.LoadTrainData(new string[1] { trainfile });
                        dnnTrain.LoadValidateData(new string[2] { validfile, testfile });
                        break;                    
                }

                if (ParameterSetting.CheckData)
                {
                    dnnTrain.CheckDataOnly();
                    return;
                }

                dnnTrain.ModelInit_FromConfig();
                dnnTrain.Training();
                dnnTrain.Dispose();
                
                log_writer.Close();
                log_stream.Close();
                if (ParameterSetting.CuBlasEnable)
                {
                    Cudalib.CUBLAS_Destroy();
                }
            }
            catch (Exception exc)
            {
                Console.Error.WriteLine(exc.ToString());
                Environment.Exit(0);
            }
        }
        
        
    }
}
