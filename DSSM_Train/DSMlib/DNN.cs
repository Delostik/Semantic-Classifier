using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace DSMlib
{
    public enum A_Func { Linear = 0, Tanh = 1, Rectified = 2 };
    public enum N_Type { Fully_Connected = 0, Convolution_layer = 1, /*Added by Ziyu Guan*/MultiWidthConv_layer = 2, Composite_Full = 3, bLSTM = 4/*Added by Ziyu Guan*/ };
    public enum P_Pooling {MAX_Pooling = 0 };
    /// <summary>
    /// Model related parameters
    /// </summary>
    public class NeuralLayer
    {
        public int Number;        
        public NeuralLayer(int num)
        {
            Number = num;            
        }        
    }    
    /// <summary>
    /// Model related parameters
    /// </summary>
    public class NeuralLink : IDisposable
    {
        public NeuralLayer Neural_In;
        public NeuralLayer Neural_Out;

        public CudaPieceFloat weight;
        public CudaPieceFloat bias;

        //***** Added by Ziyu Guan
        // for recurrent input from previous step's output; for both weight and recur_weight, we store 8 weight matrices in sequence
        // for weight: W_a, reverse_W_a, W_i, reverse_W_i, W_f, reverse_W_f, W_o, reverse_W_o, each of size (output_dim/2) * feature_dim; 
        // for recurrent weight: U_a, reverse_U_a, U_i, reverse_U_i, U_f, reverse_U_f, U_o, reverse_U_o, each of size (output_dim/2) * (output_dim/2)
        public CudaPieceFloat recur_weight;
        public CudaPieceInt winsizes = null;
        public CudaPieceInt fmsizes = null;
        public int[] wnd_sizes { get { return winsizes.MemPtr; } }
        public int[] num_fms { get { return fmsizes.MemPtr; } } // number of feature maps for each window size. This array must have the same size as wnd_sizes and the sum of its elements must equal to Neural_Out.Number
        public int[] ma_sizes;
        public NeuralLayer Extra_Input; // for extra context input, Nt = Composite_Full
        //***** Added by Ziyu Guan

        public IntPtr Weight { get { return weight.CudaPtr; } }
        public IntPtr Bias { get { return bias.CudaPtr; } }
        public IntPtr Re_weight { get { return recur_weight.CudaPtr; } } 

        public float[] Back_Weight { get { return weight.MemPtr; } }
        public float[] Back_Bias { get { return bias.MemPtr; } }
        public float[] Back_reWeight { get { return recur_weight.MemPtr; } }

        public A_Func Af;
        public N_Type Nt = N_Type.Fully_Connected;
        public int N_Winsize = 1;

        public P_Pooling pool_type = P_Pooling.MAX_Pooling;

        public float initHidBias = 0;
        public float initWeightSigma = 0.2f;

        unsafe public void CopyOutFromCuda()
        {
            weight.CopyOutFromCuda();
            bias.CopyOutFromCuda();
            if (Nt == N_Type.bLSTM)
                recur_weight.CopyOutFromCuda();
        }

        unsafe public void CopyIntoCuda()
        {
            weight.CopyIntoCuda();
            bias.CopyIntoCuda();
            if (Nt == N_Type.bLSTM)
                recur_weight.CopyIntoCuda();
        }

        //***** Modified by Ziyu Guan
        public NeuralLink(NeuralLayer layer_in, NeuralLayer layer_out, A_Func af, float hidBias, float weightSigma, N_Type nt, int win_size, bool backupOnly, int[] winsizes, int[] fmcounts, NeuralLayer extra)
        {
            Neural_In = layer_in;
            Neural_Out = layer_out;
            Extra_Input = extra;
            //Neural_In.Number = Neural_In.Number; // *N_Winsize;
            Nt = nt;
            N_Winsize = win_size;

            if (winsizes != null && fmcounts != null)
            {
                this.winsizes = new CudaPieceInt(winsizes.Length, true, true);
                this.fmsizes = new CudaPieceInt(fmcounts.Length, true, true);
                for (int i = 0; i < winsizes.Length; i++)
                {
                    this.winsizes.MemPtr[i] = winsizes[i];
                    this.fmsizes.MemPtr[i] = fmcounts[i];
                }
                this.winsizes.CopyIntoCuda();
                this.fmsizes.CopyIntoCuda();
            }

            Af = af;
            initHidBias = hidBias;
            initWeightSigma = weightSigma;
            if (Nt == N_Type.MultiWidthConv_layer)
            {
                ma_sizes = new int[num_fms.Length];
                int totalw = 0;
                for (int i = 0; i < num_fms.Length; i++)
                {
                    totalw += Neural_In.Number * wnd_sizes[i] * num_fms[i];
                    ma_sizes[i] = totalw;
                }
                weight = new CudaPieceFloat(totalw, true, backupOnly ? false : true); // for multi-window case, there are multiple matrices stored in the weight variable, each in num_fms[i] X (wnd_sizes[i] * inputDim)
            }
            else if (Nt == N_Type.Composite_Full)
            {
                weight = new CudaPieceFloat((Neural_In.Number * N_Winsize + Extra_Input.Number) * Neural_Out.Number, true, backupOnly ? false : true); // output * input, output * extra_input
            }
            else if (Nt == N_Type.bLSTM)
            {// Note that output size is double the number of dimensions of LSTM output since it is bidirectional, i.e. we have two sets of LSTMs
                weight = new CudaPieceFloat(Neural_In.Number * (Neural_Out.Number/2) * 4 * 2, true, backupOnly ? false : true);
                recur_weight = new CudaPieceFloat((Neural_Out.Number / 2) * (Neural_Out.Number / 2) * 4 * 2, true, backupOnly ? false : true);
            }
            else
                weight = new CudaPieceFloat(Neural_In.Number * Neural_Out.Number * N_Winsize, true, backupOnly ? false : true);
            
            if (Nt == N_Type.bLSTM)
                bias = new CudaPieceFloat((Neural_Out.Number/2) * 4 * 2, true, backupOnly ? false : true);
            else 
                bias = new CudaPieceFloat(Neural_Out.Number, true, backupOnly ? false : true);           
        }
        //***** Modified by Ziyu Guan

        ~NeuralLink()
        {
            Dispose();
        }


        public void Dispose()
        {
            weight.Dispose();
            bias.Dispose();
            if (recur_weight != null)
                recur_weight.Dispose();
            if (winsizes != null)
                winsizes.Dispose();
            if (fmsizes != null)
                fmsizes.Dispose();
        }

        //***** Modified by Ziyu Guan
        private void fillRand(float[] arr, int insize, int outsize, Random random)
        {
            float scale = (float)(Math.Sqrt(6.0 / (insize + outsize)) * 2);
            float bias = (float)(-Math.Sqrt(6.0 / (insize + outsize)));
            for (int i = 0; i < arr.Length; i++)
                arr[i] = (float)(random.NextDouble() * scale + bias);
        }

        public void Init()
        {
            if (Nt == N_Type.MultiWidthConv_layer)
            {
                float[] initWei = new float[weight.Size];
                Random random = ParameterSetting.Random;
                int idx = 0, accu = 0;
                int insize, outsize;
                float scale1, bias1;
                for (int i = 0; i < num_fms.Length; i++)
                {
                    insize = Neural_In.Number * wnd_sizes[i];
                    outsize = num_fms[i];
                    scale1 = (float)(Math.Sqrt(6.0 / (insize + outsize)) * 2);
                    bias1 = (float)(-Math.Sqrt(6.0 / (insize + outsize)));
                    accu += insize * outsize;
                    for (; idx < accu; idx++)
                        initWei[idx] = (float)(random.NextDouble() * scale1 + bias1);
                }
                weight.Init(initWei);
            }
            else if (Nt == N_Type.bLSTM)
            {
                // init the 8 weight matrics (each of size (outputDim/2)*inputDim) for input and 8 weight matrics for recurrent input (each of size (outputDim/2)*(outputDim/2))
                float[] initWei = new float[weight.Size];
                Random random = ParameterSetting.Random;
                int insize = Neural_In.Number;
                int outsize = Neural_Out.Number / 2;
                fillRand(initWei, insize, outsize, random);
                weight.Init(initWei);

                float[] initWei2 = new float[recur_weight.Size];
                insize = Neural_Out.Number / 2;
                outsize = Neural_Out.Number / 2;
                fillRand(initWei2, insize, outsize, random);
                recur_weight.Init(initWei2);
            }
            else
            {
                int inputsize = Neural_In.Number * N_Winsize;
                if (Nt == N_Type.Composite_Full)
                    inputsize += Extra_Input.Number;
                int outputsize = Neural_Out.Number;
                weight.Init((float)(Math.Sqrt(6.0 / (inputsize + outputsize)) * 2), (float)(-Math.Sqrt(6.0 / (inputsize + outputsize))));
            }

            bias.Init(initHidBias);
        }
        //***** Modified by Ziyu Guan

        public void Init(float wei_scale, float wei_bias)
        {
            weight.Init(wei_scale, wei_bias);
            bias.Init(initHidBias);

            if (Nt == N_Type.bLSTM)
                recur_weight.Init(wei_scale, wei_bias);
        }

        public void Init(NeuralLink refLink)
        {
            weight.Init(refLink.Back_Weight);
            bias.Init(refLink.Back_Bias);

            if (Nt == N_Type.bLSTM)
                recur_weight.Init(refLink.Back_reWeight);
        }        
    }

    /// <summary>
    /// Lookup tables contain trainable word vectors or product feature context vectors
    /// </summary>
    public class LookupTab : IDisposable
    {
        public CudaPieceFloat table;

        public int vecDim;
        public int count;

        public IntPtr LookupTable { get { return table.CudaPtr; } }
        public float[] Back_LookupTable { get { return table.MemPtr; } }

        unsafe public void CopyOutFromCuda()
        {
            table.CopyOutFromCuda();
        }

        unsafe public void CopyIntoCuda()
        {
            table.CopyIntoCuda();
        }

        public LookupTab(int vecDim, int count, bool backupOnly)
        {
            this.vecDim = vecDim;
            this.count = count;
            table = new CudaPieceFloat(vecDim * count, true, backupOnly ? false : true);
        }

        ~LookupTab()
        {
            Dispose();
        }

        public void Dispose()
        {
            table.Dispose();
        }

        public void Init()
        {
            table.Init("lookupt", vecDim);
        }

        public void Init(string wvfile)
        {
            FileStream mstream = new FileStream(wvfile, FileMode.Open, FileAccess.Read);
            BinaryReader mreader = new BinaryReader(mstream);
            int wordnum = mreader.ReadInt32();
            int dim = mreader.ReadInt32();
            if (wordnum != this.count || dim != this.vecDim)
            {
                mreader.Close();
                mstream.Close();
                throw new Exception("Inconsistent word number or word vector dimension encountered: word num" + wordnum.ToString());
            }
            int ltlength = dim * wordnum;
            for (int mm = 0; mm < ltlength; mm++)
                Back_LookupTable[mm] = (float)mreader.ReadDouble();
            table.CopyIntoCuda();

            mreader.Close();
            mstream.Close();
        }

        public void Init(float wei_scale, float wei_bias)
        {
            table.Init(wei_scale, wei_bias);
        }

        public void Init(LookupTab refTable)
        {
            table.Init(refTable.Back_LookupTable);
        }
    }

    /// <summary>
    /// Model related parameters and network structure
    /// </summary>
    public class DNN
    {
        public List<NeuralLayer> neurallayers = new List<NeuralLayer>();
        public List<NeuralLink> neurallinks = new List<NeuralLink>();

        public LookupTab wordLT, contextLT;

        public DNN(string fileName)
        {
            Model_Load(fileName, true);
        }

        public int OutputLayerSize
        {
            get { return neurallayers.Last().Number; }
        }

        public int ModelParameterNumber
        {
            get
            {
                int NUM = 0;

                // Count parameters in lookup tables
                NUM += wordLT.vecDim * wordLT.count;
                NUM += contextLT.vecDim * contextLT.count;

                for (int i = 0; i < neurallinks.Count; i++)
                {
                    int num = 0;
                    if (neurallinks[i].Nt == N_Type.Fully_Connected || neurallinks[i].Nt == N_Type.Convolution_layer)
                    {
                        num += neurallinks[i].Neural_In.Number * neurallinks[i].N_Winsize * neurallinks[i].Neural_Out.Number;
                    }
                    else if (neurallinks[i].Nt == N_Type.MultiWidthConv_layer)
                    {
                        for (int j = 0; j < neurallinks[i].num_fms.Length; j++)
                            num += neurallinks[i].Neural_In.Number * neurallinks[i].wnd_sizes[j] * neurallinks[i].num_fms[j];
                    }
                    else if (neurallinks[i].Nt == N_Type.bLSTM)
                    {
                        num += neurallinks[i].Neural_In.Number * (neurallinks[i].Neural_Out.Number / 2) * 4 * 2;
                        num += (neurallinks[i].Neural_Out.Number / 2) * (neurallinks[i].Neural_Out.Number / 2) * 4 * 2;
                    }
                    else // for Composite Full layer
                    {
                        num += (neurallinks[i].Neural_In.Number * neurallinks[i].N_Winsize + neurallinks[i].Extra_Input.Number) * neurallinks[i].Neural_Out.Number;
                    }

                    if (ParameterSetting.UpdateBias)
                    {
                        if (neurallinks[i].Nt == N_Type.bLSTM)
                            num += (neurallinks[i].Neural_Out.Number / 2) * 4 * 2;
                        else
                            num += neurallinks[i].Neural_Out.Number;
                    }

                    NUM += num;
                }
                return NUM;
            }
        }

        public void CopyOutFromCuda()
        {
            wordLT.CopyOutFromCuda();
            contextLT.CopyOutFromCuda();
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].CopyOutFromCuda();
            }
        }

        public void CopyIntoCuda()
        {
            wordLT.CopyIntoCuda();
            contextLT.CopyIntoCuda();
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].CopyIntoCuda();
            }
        }
        /// <summary>
        /// For backward-compatible, neurallinks[i].Af = tanh is stored as 0, neurallinks[i].Af = linear is stored as 1, neurallinks[i].Af = rectified is stored as 2 
        /// Do not alter the ordering of those existing A_Func elements.
        /// </summary>
        A_Func[] Int2A_FuncMapping = new A_Func[] {DSMlib.A_Func.Tanh, DSMlib.A_Func.Linear, DSMlib.A_Func.Rectified};
        
        public int A_Func2Int(A_Func af)
        {
            for(int i = 0; i < Int2A_FuncMapping.Length; ++i)
            {
                if(Int2A_FuncMapping[i] == af)
                { 
                    return i; 
                }
            }
            return 0;
        }        
        public A_Func Int2A_Func(int af)
        {
            return Int2A_FuncMapping[af];            
        }
        public void Model_Save(string fileName)
        {
            FileStream mstream = new FileStream(fileName, FileMode.Create, FileAccess.Write);
            BinaryWriter mwriter = new BinaryWriter(mstream);
            mwriter.Write(neurallayers.Count);
            for (int i = 0; i < neurallayers.Count; i++)
            {
                mwriter.Write(neurallayers[i].Number);
            }
            mwriter.Write(neurallinks.Count);
            for (int i = 0; i < neurallinks.Count; i++)
            {              
                //// compose a Int32 integer whose higher 16 bits store activiation function and lower 16 bits store network type
                //// In addition, for backward-compatible, neurallinks[i].Af = tanh is stored as 0, neurallinks[i].Af = linear is stored as 1, neurallinks[i].Af = rectified is stored as 2 
                //// Refer to the Int2A_FuncMapping                
                int afAndNt = ( A_Func2Int(neurallinks[i].Af) << 16) | ((int) neurallinks[i].Nt );
                mwriter.Write(afAndNt);
                mwriter.Write((int)neurallinks[i].pool_type);
                if (neurallinks[i].Nt == N_Type.MultiWidthConv_layer)
                {
                    mwriter.Write(neurallinks[i].num_fms.Length);
                    for (int j = 0; j < neurallinks[i].num_fms.Length; j++)
                    {
                        mwriter.Write(neurallinks[i].num_fms[j]);
                        mwriter.Write(neurallinks[i].wnd_sizes[j]);
                    }
                } else if (neurallinks[i].Nt == N_Type.Composite_Full)
                {
                    mwriter.Write(neurallinks[i].Extra_Input.Number);
                }
                mwriter.Write(neurallinks[i].Neural_In.Number);
                mwriter.Write(neurallinks[i].Neural_Out.Number);
                mwriter.Write(neurallinks[i].initHidBias);
                mwriter.Write(neurallinks[i].initWeightSigma);
                mwriter.Write(neurallinks[i].N_Winsize);
            }

            for (int i = 0; i < neurallinks.Count; i++)
            {
                mwriter.Write(neurallinks[i].Back_Weight.Length);
                for (int m = 0; m < neurallinks[i].Back_Weight.Length; m++)
                {
                    mwriter.Write(neurallinks[i].Back_Weight[m]);
                }

                if (neurallinks[i].Nt == N_Type.bLSTM)
                {
                    //store recurrent weights
                    mwriter.Write(neurallinks[i].Back_reWeight.Length);
                    for (int kk = 0; kk < neurallinks[i].Back_reWeight.Length; kk++)
                    {
                        mwriter.Write(neurallinks[i].Back_reWeight[kk]);
                    }
                    //store bias
                    mwriter.Write(neurallinks[i].Back_Bias.Length);
                    for (int kk = 0; kk < neurallinks[i].Back_Bias.Length; kk++)
                    {
                        mwriter.Write(neurallinks[i].Back_Bias[kk]);
                    }
                } 
                else
                {
                    mwriter.Write(neurallinks[i].Neural_Out.Number);
                    for (int m = 0; m < neurallinks[i].Neural_Out.Number; m++)
                    {
                        mwriter.Write(neurallinks[i].Back_Bias[m]);
                    }
                }
                
            }

            //finally write Lookup table
            mwriter.Write(wordLT.vecDim);
            mwriter.Write(wordLT.count);
            for (int m = 0; m < wordLT.Back_LookupTable.Length; m++)
                mwriter.Write(wordLT.Back_LookupTable[m]);

            mwriter.Write(contextLT.vecDim);
            mwriter.Write(contextLT.count);
            for (int m = 0; m < contextLT.Back_LookupTable.Length; m++)
                mwriter.Write(contextLT.Back_LookupTable[m]);

            mwriter.Close();
            mstream.Close();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fileName"></param>
        /// <param name="allocateStructureFromEmpty">True will init DNN structure and allocate new space; False will only load data from file</param>
        public void Model_Load(string fileName, bool allocateStructureFromEmpty)
        {
            FileStream mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
            BinaryReader mreader = new BinaryReader(mstream);

            List<int> layer_info = new List<int>();
            int mlayer_num = mreader.ReadInt32();
            for (int i = 0; i < mlayer_num; i++)
            {
                layer_info.Add(mreader.ReadInt32());
            }
            if (allocateStructureFromEmpty)
            {
                neurallayers.Clear();
                for (int i = 0; i < layer_info.Count; i++)
                {
                    NeuralLayer layer = new NeuralLayer(layer_info[i]);
                    neurallayers.Add(layer);
                }
            }

            int mlink_num = mreader.ReadInt32();
            if (allocateStructureFromEmpty)
                neurallinks.Clear();
            for (int i = 0; i < mlink_num; i++)
            {
                int afAndNt = mreader.ReadInt32();
                A_Func aF = Int2A_Func(afAndNt >> 16);
                N_Type mnt = (N_Type)(afAndNt & ((1 << 16) - 1));
                P_Pooling mp = (P_Pooling)mreader.ReadInt32();
                int[] wnds = null, fms = null;
                int numofWnd;
                NeuralLayer extraLayer = null;
                if (mnt == N_Type.MultiWidthConv_layer)
                {
                    numofWnd = mreader.ReadInt32();
                    wnds = new int[numofWnd];
                    fms = new int[numofWnd];
                    for (int jj = 0; jj < fms.Length; jj++)
                    {
                        fms[jj] = mreader.ReadInt32();
                        wnds[jj] = mreader.ReadInt32();
                    }
                }
                else if (mnt == N_Type.Composite_Full)
                {
                    extraLayer = new NeuralLayer(mreader.ReadInt32());
                }

                int in_num = mreader.ReadInt32();
                int out_num = mreader.ReadInt32();
                float inithidbias = mreader.ReadSingle();
                float initweightsigma = mreader.ReadSingle();
                int mws = mreader.ReadInt32();

                //if (ParameterSetting.LoadModelOldFormat) delete this option

                if (allocateStructureFromEmpty)
                {
                    NeuralLink link = new NeuralLink(neurallayers[i], neurallayers[i + 1], aF, 0, initweightsigma, mnt, mws, false, wnds, fms, extraLayer);
                    neurallinks.Add(link);
                }
            }

            for (int i = 0; i < mlink_num; i++)
            {
                int weight_len = mreader.ReadInt32(); // Write(neurallinks[i].Back_Weight.Length);
                if (weight_len != neurallinks[i].Back_Weight.Length)
                {
                    Console.WriteLine("Loading Model Weight Error on layer" + i.ToString() +"!  " + weight_len.ToString() + " " + neurallinks[i].Back_Weight.Length.ToString());
                    Console.ReadLine();
                }
                for (int m = 0; m < weight_len; m++)
                {
                    neurallinks[i].Back_Weight[m] = mreader.ReadSingle();
                }

                if (neurallinks[i].Nt == N_Type.bLSTM)
                {
                    int recuWeight_len = mreader.ReadInt32();
                    if (recuWeight_len != neurallinks[i].Back_reWeight.Length)
                    {
                        Console.WriteLine("Loading Model Recurrent Weight Error on layer" + i.ToString() + "!  " + recuWeight_len.ToString() + " " + neurallinks[i].Back_reWeight.Length.ToString());
                        Console.ReadLine();
                    }
                    for (int m = 0; m < recuWeight_len; m++)
                    {
                        neurallinks[i].Back_reWeight[m] = mreader.ReadSingle();
                    }
                }
                int bias_len = mreader.ReadInt32();
                if (bias_len != neurallinks[i].Back_Bias.Length)
                {
                    Console.WriteLine("Loading Model Bias Error on layer" + i.ToString() + "!  " + bias_len.ToString() + " " + neurallinks[i].Back_Bias.Length.ToString());
                    Console.ReadLine();
                }
                for (int m = 0; m < bias_len; m++)
                {
                    neurallinks[i].Back_Bias[m] = mreader.ReadSingle();
                }
            }

            //load lookup tables
            Console.WriteLine("Loading lookup tables...");
            int wordVeclen = mreader.ReadInt32();
            int wordCount = mreader.ReadInt32();
            if (allocateStructureFromEmpty)
                wordLT = new LookupTab(wordVeclen, wordCount, false);
            int ltlength = wordVeclen * wordCount;
            if (ltlength != wordLT.Back_LookupTable.Length)
            {
                Console.WriteLine("Loading Word Lookup Table Error!  " + ltlength.ToString() + " " + wordLT.Back_LookupTable.Length.ToString());
                Console.ReadLine();
            }
            for (int mm = 0; mm < ltlength; mm++)
                wordLT.Back_LookupTable[mm] = mreader.ReadSingle();

            int contextVeclen = mreader.ReadInt32();
            int contextCount = mreader.ReadInt32();
            if (allocateStructureFromEmpty)
                contextLT = new LookupTab(contextVeclen, contextCount, false);
            ltlength = contextCount * contextVeclen;
            if (ltlength != contextLT.Back_LookupTable.Length)
            {
                Console.WriteLine("Loading Context Lookup Table Error!  " + ltlength.ToString() + " " + contextLT.Back_LookupTable.Length.ToString());
                Console.ReadLine();
            }
            for (int mm = 0; mm < ltlength; mm++)
                contextLT.Back_LookupTable[mm] = mreader.ReadSingle();

            mreader.Close();
            mstream.Close();
            CopyIntoCuda();
        }

        //public void Fill_Layer_One(string fileName)
        //{
        //    FileStream mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
        //    BinaryReader mreader = new BinaryReader(mstream);

        //    List<int> layer_info = new List<int>();
        //    int mlayer_num = mreader.ReadInt32();
        //    for (int i = 0; i < mlayer_num; i++)
        //    {
        //        layer_info.Add(mreader.ReadInt32());
        //    }

        //    //for (int i = 0; i < layer_info.Count; i++)
        //    //{
        //    //    NeuralLayer layer = new NeuralLayer(layer_info[i]);
        //    //    neurallayers.Add(layer);
        //    //}

        //    int mlink_num = mreader.ReadInt32();
        //    for (int i = 0; i < mlink_num; i++)
        //    {
        //        int in_num = mreader.ReadInt32();
        //        int out_num = mreader.ReadInt32();
        //        float inithidbias = mreader.ReadSingle();
        //        float initweightsigma = mreader.ReadSingle();
        //        int mws = mreader.ReadInt32();
        //        N_Type mnt = (N_Type)mreader.ReadInt32();
        //        P_Pooling mp = (P_Pooling)mreader.ReadInt32();

        //        //NeuralLink link = new NeuralLink(neurallayers[i], neurallayers[i + 1], A_Func.Tanh, 0, initweightsigma,mnt,mws);
        //        //neurallinks.Add(link);
        //    }

        //    for (int i = 0; i < mlink_num; i++)
        //    {
        //        int weight_len = mreader.ReadInt32(); // Write(neurallinks[i].Back_Weight.Length);
        //        if (weight_len != neurallinks[i].Back_Weight.Length)
        //        {
        //            Console.WriteLine("Loading Model Weight Error!  " + weight_len.ToString() + " " + neurallinks[i].Back_Weight.Length.ToString());
        //            Console.ReadLine();
        //        }
        //        for (int m = 0; m < weight_len; m++)
        //        {
        //            neurallinks[i].Back_Weight[m] = mreader.ReadSingle();
        //        }
        //        int bias_len = mreader.ReadInt32();
        //        if (bias_len != neurallinks[i].Back_Bias.Length)
        //        {
        //            Console.WriteLine("Loading Model Bias Error!  " + bias_len.ToString() + " " + neurallinks[i].Back_Bias.Length.ToString());
        //            Console.ReadLine();
        //        }
        //        for (int m = 0; m < bias_len; m++)
        //        {
        //            neurallinks[i].Back_Bias[m] = mreader.ReadSingle();
        //        }
        //    }
        //    mreader.Close();
        //    mstream.Close();

        //    for (int i = 1; i < neurallinks.Count; i++)
        //    {
        //        for (int m = 0; m < neurallinks[i].Back_Bias.Length; m++)
        //        {
        //            neurallinks[i].Back_Bias[m] = 0;
        //        }
        //        int wei_num = neurallinks[i].Back_Weight.Length;
        //        for (int m = 0; m < neurallinks[i].Neural_Out.Number; m++)
        //        {
        //            neurallinks[i].Back_Weight[(m * neurallinks[i].Neural_Out.Number) % wei_num + m] = 1.0f;
        //        }
        //    }

        //    CopyIntoCuda();
        //}

        public string DNN_Descr()
        {
            string result = "";
            for (int i = 0; i < neurallayers.Count; i++)
            {
                result += "Neural Layer " + i.ToString() + ": " + neurallayers[i].Number.ToString() + "\n";
            }

            for (int i = 0; i < neurallinks.Count; i++)
            {
                result += "layer " + i.ToString() + " to layer " + (i + 1).ToString() + ":" +
                        " Neural Type : " + neurallinks[i].Nt.ToString() + ";" +
                        " AF Type : " + neurallinks[i].Af.ToString() + ";" +
                        " hid bias : " + neurallinks[i].initHidBias.ToString() + ";" +
                        " weight sigma : " + neurallinks[i].initWeightSigma.ToString() + ";";
                if (neurallinks[i].Nt == N_Type.MultiWidthConv_layer)
                {
                    string temp1 = "", temp2 = "";
                    for (int j = 0; j < neurallinks[i].wnd_sizes.Length; j++)
                    {
                        if (j == 0)
                        {
                            temp1 += "(" + neurallinks[i].wnd_sizes[j].ToString();
                            temp2 += "(" + neurallinks[i].num_fms[j].ToString();
                        }
                        else if(j == neurallinks[i].wnd_sizes.Length-1)
                        {
                            temp1 += ", " + neurallinks[i].wnd_sizes[j].ToString() + ")";
                            temp2 += ", " + neurallinks[i].num_fms[j].ToString() + ")";
                        }
                        else
                        {
                            temp1 += ", " + neurallinks[i].wnd_sizes[j].ToString();
                            temp2 += ", " + neurallinks[i].num_fms[j].ToString();
                        }
                    }
                    result += " Window Sizes : " + temp1 + ";" + " Feature Map Sizes : " + temp2 + ";";
                    result += " Pooling Type : " + neurallinks[i].pool_type.ToString() + ";" + "\n";
                }
                else if (neurallinks[i].Nt == N_Type.Convolution_layer)
                {
                    result += " Window Size : " + neurallinks[i].N_Winsize.ToString() + ";";
                    result += " Pooling Type : " + neurallinks[i].pool_type.ToString() + ";" + "\n";
                }
                else if (neurallinks[i].Nt == N_Type.bLSTM)
                {
                    result += " Pooling Type : " + neurallinks[i].pool_type.ToString() + ";" + "\n";
                }
                else if (neurallinks[i].Nt == N_Type.Composite_Full)
                {
                    result += " Context Input Size : " + neurallinks[i].Extra_Input.Number.ToString() + ";" + "\n";
                }                        
            }

            //For lookup tables
            result += "Word Lookup Table Size : " + wordLT.vecDim + " X " + wordLT.count + "\n";
            result += "Context Lookup Table Size : " + contextLT.vecDim + " X " + contextLT.count + "\n";

            return result;
        }

        public DNN(int featureSize, int[] layerDim, int[] activation, float[] sigma, int[] arch, int[] wind, int contextDim, int wordNum, int contextNum, int[] wndN, int[] fmN, bool backupOnly)
        {
            NeuralLayer inputlayer = new NeuralLayer(featureSize);
            neurallayers.Add(inputlayer);
            for (int i = 0; i < layerDim.Length; i++)
            {
                NeuralLayer layer = new NeuralLayer(layerDim[i]);
                neurallayers.Add(layer);
            }

            for (int i = 0; i < layerDim.Length; i++)
            {
                NeuralLink link = null;
                N_Type tp = (N_Type)arch[i];
                if (tp == N_Type.MultiWidthConv_layer)
                {
                    link = new NeuralLink(neurallayers[i], neurallayers[i + 1], (A_Func)activation[i], 0, sigma[i], tp, 0, backupOnly, wndN, fmN, null);
                }
                else if (tp == N_Type.Composite_Full)
                {
                    NeuralLayer extrain = new NeuralLayer(contextDim);
                    link = new NeuralLink(neurallayers[i], neurallayers[i + 1], (A_Func)activation[i], 0, sigma[i], tp, wind[i], backupOnly, null, null, extrain);
                }
                else
                    link = new NeuralLink(neurallayers[i], neurallayers[i + 1], (A_Func)activation[i], 0, sigma[i], tp, wind[i], backupOnly, null, null, null);
                neurallinks.Add(link);
            }

            // Construct LT
            wordLT = new LookupTab(featureSize, wordNum, backupOnly);
            contextLT = new LookupTab(contextDim, contextNum, backupOnly);
        }
        
        public void Init()
        {
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].Init();
            }
            if (ParameterSetting.WORDLT_INIT == null || !File.Exists(ParameterSetting.WORDLT_INIT))
                wordLT.Init();
            else
            {
                Program.Print("=========Using Google word vector to init==========");
                wordLT.Init(ParameterSetting.WORDLT_INIT);
            }

            contextLT.Init();
        }

        public void Init(DNN model)
        {
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].Init(model.neurallinks[i]);
            }
            wordLT.Init(model.wordLT);
            contextLT.Init(model.contextLT);
        }

        public void Init(float wei_scale, float wei_bias)
        {
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].Init(wei_scale, wei_bias);
            }
            wordLT.Init(wei_scale, wei_bias);
            contextLT.Init(wei_scale, wei_bias);
        }

        /// <summary>
        /// Before call this stuff, you must call CopyOutFromCuda()
        /// The returns is only used for backup purpose. So its does not allocate any GPU memory.
        /// </summary>
        /// <returns></returns>
        public DNN CreateBackupClone()
        {
            // Assumen the first link layer is the multi-convolutional layer
            DNN backupClone = new DNN(
                this.neurallayers[0].Number,
                this.neurallinks.Select(o => o.Neural_Out.Number).ToArray(),
                this.neurallinks.Select(o => (int)o.Af).ToArray(),
                this.neurallinks.Select(o => o.initWeightSigma).ToArray(),
                this.neurallinks.Select(o => (int)o.Nt).ToArray(),
                this.neurallinks.Select(o => o.N_Winsize).ToArray(),
                this.contextLT.vecDim,
                this.wordLT.count,
                this.contextLT.count,
                this.neurallinks[0].wnd_sizes,
                this.neurallinks[0].num_fms,
                true);
            backupClone.Init(this);
            return backupClone;
        }
    }

    public class LookupTabRunData : IDisposable
    {
        LookupTab table;
        public LookupTab Table { get { return table; } }

        public int Dim { get { return table.vecDim; } }
        public int Count { get { return table.count; } }
        bool isWordInput; // if wordInput, use MAXSEGMENT_BATCH to construct InputDeriv

        CudaPieceFloat[] inputDerivs = null;
        CudaPieceFloat tabUpdate = null;
        CudaPieceFloat tabAdaGrad = null;

        public CudaPieceFloat[] InputDeriv { get { return inputDerivs; } }
        public CudaPieceFloat TabUpdate { get { return tabUpdate; } }
        public CudaPieceFloat TabAdaGrad { get { return tabAdaGrad; } }

        //some auxiliary variables
        public CudaPieceInt uniqueWordID = null;
        public CudaPieceInt uniqueWordIdx = null;
        public CudaPieceInt Sequence = null;
        public int uniqueNum;

        public LookupTabRunData(LookupTab table, bool isWordInput)
        {
            this.table = table;
            this.isWordInput = isWordInput;
            
            inputDerivs = new CudaPieceFloat[3];
            if (ParameterSetting.updateScheme == 1 || ParameterSetting.CheckGrad)
                tabUpdate = new CudaPieceFloat(Dim * Count, ParameterSetting.CheckGrad ? true : false, true);
            else if (ParameterSetting.updateScheme == 2)
                tabAdaGrad = new CudaPieceFloat(Dim * Count, false, true);
            if (isWordInput)
            {
                for (int i = 0; i < inputDerivs.Length; i++)
                    inputDerivs[i] = new CudaPieceFloat(Dim * PairInputStream.MAXSEGMENT_BATCH, false, true);
                Sequence = new CudaPieceInt(3 * PairInputStream.MAXSEGMENT_BATCH, true, true);
            }
            else
            {
                for (int i = 0; i < inputDerivs.Length; i++)
                    inputDerivs[i] = new CudaPieceFloat(Dim * ParameterSetting.BATCH_SIZE, false, true);
                Sequence = new CudaPieceInt(3 * ParameterSetting.BATCH_SIZE, true, true);
            }
            uniqueWordID = new CudaPieceInt(Count, true, true);
            uniqueWordIdx = new CudaPieceInt(Count, true, true);
            
        }

        ~LookupTabRunData()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (inputDerivs != null)
            {
                foreach (CudaPieceFloat i in inputDerivs)
                    i.Dispose();
                inputDerivs = null;
            }
            if (tabUpdate != null)
            {
                tabUpdate.Dispose();
                tabUpdate = null;
            }
            if (uniqueWordID != null)
                uniqueWordID.Dispose();
            if (uniqueWordIdx != null)
                uniqueWordIdx.Dispose();
            if (Sequence != null)
                Sequence.Dispose();
            if (tabAdaGrad != null)
                tabAdaGrad.Dispose();
        }

        public void ZeroDeriv()
        {
            if (inputDerivs != null)
            {
                foreach (CudaPieceFloat e in inputDerivs)
                    e.Zero();
            }
        }

    }

    /// <summary>
    /// A particular run related data, including feedforward outputs and backward propagation derivatives
    /// </summary>
    public class NeuralLayerData : IDisposable
    {
        NeuralLayer LayerModel;
        public int Number { get { return LayerModel.Number; } }
        /// <summary>        
        /// </summary>
        /// <param name="num"></param>
        /// <param name="isValueNeeded">To save GPU memory, when no errors are needed, we should not allocate error piece. This usually happens on the input layer</param>
        public NeuralLayerData(NeuralLayer layerModel, bool isValueNeeded)
        {
            LayerModel = layerModel;
            if (isValueNeeded)
            {
                outputs = new CudaPieceFloat[3];
                errorDerivs = new CudaPieceFloat[3];
                for (int i = 0; i < outputs.Length; i++)
                {
                    outputs[i] = new CudaPieceFloat(ParameterSetting.BATCH_SIZE * Number, ParameterSetting.DEBUG ? true : false, true);
                    errorDerivs[i] = new CudaPieceFloat(ParameterSetting.BATCH_SIZE * Number, ParameterSetting.DEBUG ? true : false, true);                   
                }
            }
        }
        /// <summary>
        /// The output of the layer, i.e., the actual activitation values
        /// </summary>
        CudaPieceFloat[] outputs = null;

        public CudaPieceFloat[] Outputs
        {
            get { return outputs; }
        }
        /// <summary>
        /// The error of the layer, back-propagated from the top loss function
        /// </summary>
        CudaPieceFloat[] errorDerivs = null;

        public CudaPieceFloat[] ErrorDerivs
        {
            get { return errorDerivs; }
        }


        ~NeuralLayerData()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (outputs != null)
            {
                foreach (CudaPieceFloat e in outputs)
                    e.Dispose();
                outputs = null;
            }
            if (errorDerivs != null)
            {
                foreach (CudaPieceFloat e in errorDerivs)
                    e.Dispose();
                errorDerivs = null;
            }
        }
    }
    /// <summary>
    /// A particular run related data, including feedforward outputs and backward propagation derivatives
    /// </summary>
    public class NeuralLinkData :IDisposable
    {
        NeuralLink neuralLinkModel;

        public NeuralLink NeuralLinkModel
        {
            get { return neuralLinkModel; }
        }
        
        /// <summary>
        /// Used if convolutional
        /// </summary>
        CudaPieceFloat[] layerPoolingOutputs = null;

        // for LSTM , this stores output h_t for each time step, of dimension (2*output_dim) * batchsize
        public CudaPieceFloat[] LayerPoolingOutputs
        {
            get { return layerPoolingOutputs; }
        }
        /// <summary>
        /// Used if convolutional and maxpooling
        /// </summary>
        CudaPieceInt[] layerMaxPooling_Indices = null;

        public CudaPieceInt[] LayerMaxPooling_Indices
        {
            get { return layerMaxPooling_Indices; }
        }

        // for LSTM exclusively, storing internal state c (forward) or derivative of c (backward)
        CudaPieceFloat[] layerPoolingInternalState = null;
        public CudaPieceFloat[] LayerPoolingInternalState
        {
            get { return layerPoolingInternalState; }
        }

        // for LSTM exclusively, storing activations(forward)/derivatives(backward) of internal input a, a =  tanh(W_a *x + U_a * h + b_a)
        CudaPieceFloat[] layerPoolingA = null;
        public CudaPieceFloat[] LayerPoolingA
        {
            get { return layerPoolingA; }
        }

        // for LSTM exclusively, storing activations(forward)/derivatives(backward) of internal input gates
        CudaPieceFloat[] layerPoolingI = null;
        public CudaPieceFloat[] LayerPoolingI
        {
            get { return layerPoolingI; }
        }

        // for LSTM exclusively, storing activations(forward)/derivatives(backward) of internal forget gates
        CudaPieceFloat[] layerPoolingF = null;
        public CudaPieceFloat[] LayerPoolingF
        {
            get { return layerPoolingF; }
        }

        // for LSTM exclusively, storing activations(forward)/derivatives(backward) of internal output gates
        CudaPieceFloat[] layerPoolingO = null;
        public CudaPieceFloat[] LayerPoolingO
        {
            get { return layerPoolingO; }
        }

        CudaPieceFloat weightDeriv = null;

        public CudaPieceFloat WeightDeriv
        {
            get { return weightDeriv; }
        }

        CudaPieceFloat reweightDeriv = null;
        public CudaPieceFloat reWeightDeriv
        {
            get { return reweightDeriv; }
        }


        CudaPieceFloat biasDeriv = null;

        public CudaPieceFloat BiasDeriv
        {
            get { return biasDeriv; }
        }

        /// <summary>
        /// Output cache for composite layer
        /// </summary>
        CudaPieceFloat[] composite_outputs = null;

        public CudaPieceFloat[] CompOutputs
        {
            get { return composite_outputs; }
        }

        /// <summary>
        /// composite error derivatives
        /// </summary>
        CudaPieceFloat[] composite_errors = null;

        public CudaPieceFloat[] CompErrors
        {
            get { return composite_errors; }
        }

        /// <summary>
        /// Wei_Update = momentum * Wei_Update + learn_rate * grad.
        /// Weight = Weight + Wei_Update
        /// </summary>
        CudaPieceFloat weightUpdate = null;

        public CudaPieceFloat WeightUpdate
        {
            get { return weightUpdate; }
        }

        CudaPieceFloat reweightUpdate = null;
        public CudaPieceFloat reWeightUpdate
        {
            get { return reweightUpdate; }
        }


        CudaPieceFloat biasUpdate = null;

        public CudaPieceFloat BiasUpdate
        {
            get { return biasUpdate; }
        }

        // used by adaptive gradient descent  adagrad += grad^2;    weight = weight - lr*grad/sqrt(adagrad)
        CudaPieceFloat weightAdaGrad = null;

        public CudaPieceFloat WeightAdaGrad
        {
            get { return weightAdaGrad; }
        }

        CudaPieceFloat reweightAdaGrad = null;

        public CudaPieceFloat reWeightAdaGrad
        {
            get { return reweightAdaGrad; }
        }


        CudaPieceFloat biasAdaGrad = null;

        public CudaPieceFloat BiasAdaGrad
        {
            get { return biasAdaGrad; }
        }

        public NeuralLinkData(NeuralLink neuralLink)
        {
            neuralLinkModel = neuralLink;
            
            if (neuralLinkModel.Nt == N_Type.Convolution_layer || neuralLinkModel.Nt == N_Type.MultiWidthConv_layer || neuralLinkModel.Nt == N_Type.bLSTM)
            {
                layerPoolingOutputs = new CudaPieceFloat[3];
                layerMaxPooling_Indices = new CudaPieceInt[3];
                for (int i = 0; i < 3; i++)
                {
                    // **now has the same shape as in single convolution, but some cells are invalid** for multi-window case, we still allocate the same size of space, but pooling output stores three matrices, each of size (seg_size - batchsize*(win_size-1))* fm_size
                    layerPoolingOutputs[i] = new CudaPieceFloat(PairInputStream.MAXSEGMENT_BATCH * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG ? true : false, true);
                    layerMaxPooling_Indices[i] = new CudaPieceInt(ParameterSetting.BATCH_SIZE * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG ? true : false, true);
                }
                
                if (neuralLinkModel.Nt == N_Type.bLSTM)
                {
                    layerPoolingInternalState = new CudaPieceFloat[3];
                    layerPoolingA = new CudaPieceFloat[3];
                    layerPoolingI = new CudaPieceFloat[3];
                    layerPoolingF = new CudaPieceFloat[3];
                    layerPoolingO = new CudaPieceFloat[3];
                    // initialize all pooling variables dedicated to LSTM
                    for (int i = 0; i < 3; i++)
                    {
                        layerPoolingInternalState[i] = new CudaPieceFloat(PairInputStream.MAXSEGMENT_BATCH * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG ? true : false, true);
                        layerPoolingA[i] = new CudaPieceFloat(PairInputStream.MAXSEGMENT_BATCH * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG ? true : false, true);
                        layerPoolingI[i] = new CudaPieceFloat(PairInputStream.MAXSEGMENT_BATCH * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG ? true : false, true);
                        layerPoolingF[i] = new CudaPieceFloat(PairInputStream.MAXSEGMENT_BATCH * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG ? true : false, true);
                        layerPoolingO[i] = new CudaPieceFloat(PairInputStream.MAXSEGMENT_BATCH * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG ? true : false, true);
                    }
                    
                }
            }

            int totalweightsize = 0;
            int totalreweightsize = 0;
            int totalbiassize = neuralLinkModel.Neural_Out.Number;
            if (neuralLinkModel.Nt == N_Type.MultiWidthConv_layer)
            {
                for (int i = 0; i < neuralLinkModel.wnd_sizes.Length; i++)
                    totalweightsize += neuralLinkModel.Neural_In.Number * neuralLinkModel.wnd_sizes[i] * neuralLinkModel.num_fms[i];
            }
            else if (neuralLinkModel.Nt == N_Type.Composite_Full)
            {
                totalweightsize = neuralLinkModel.Neural_Out.Number * (neuralLinkModel.Neural_In.Number + neuralLinkModel.Extra_Input.Number);
                // Create space for composite outputs and their derivatives
                composite_outputs = new CudaPieceFloat[3];
                composite_errors = new CudaPieceFloat[3];
                for (int i = 0; i < composite_errors.Length; i++)
                {
                    composite_outputs[i] = new CudaPieceFloat((neuralLinkModel.Neural_In.Number * neuralLinkModel.N_Winsize + neuralLinkModel.Extra_Input.Number) * ParameterSetting.BATCH_SIZE, ParameterSetting.DEBUG ? true : false, true);
                    composite_errors[i] = new CudaPieceFloat((neuralLinkModel.Neural_In.Number * neuralLinkModel.N_Winsize + neuralLinkModel.Extra_Input.Number) * ParameterSetting.BATCH_SIZE, ParameterSetting.DEBUG ? true : false, true);
                }
            }
            else if (neuralLinkModel.Nt == N_Type.bLSTM)
            {
                // calculate the size of weight and recurrent weight
                totalweightsize = neuralLinkModel.Neural_In.Number * (neuralLinkModel.Neural_Out.Number / 2) * 4 * 2;
                totalreweightsize = (neuralLinkModel.Neural_Out.Number / 2) * (neuralLinkModel.Neural_Out.Number / 2) * 4 * 2;
                totalbiassize = (neuralLinkModel.Neural_Out.Number / 2) * 4 * 2;
            }
            else
                totalweightsize = neuralLinkModel.Neural_In.Number * neuralLinkModel.Neural_Out.Number * neuralLinkModel.N_Winsize;

            weightDeriv = new CudaPieceFloat(totalweightsize, ParameterSetting.CheckGrad ? true : false, true);
            biasDeriv = new CudaPieceFloat(totalbiassize, ParameterSetting.CheckGrad ? true : false, true);
            if (neuralLinkModel.Nt == N_Type.bLSTM)
                reweightDeriv = new CudaPieceFloat(totalreweightsize, ParameterSetting.CheckGrad ? true : false, true);

            if (ParameterSetting.updateScheme == 1)
            {
                weightUpdate = new CudaPieceFloat(totalweightsize, false, true);
                biasUpdate = new CudaPieceFloat(totalbiassize, false, true);
                if (neuralLinkModel.Nt == N_Type.bLSTM)
                    reweightUpdate = new CudaPieceFloat(totalreweightsize, false, true);
            }
            else if (ParameterSetting.updateScheme == 2)
            {
                weightAdaGrad = new CudaPieceFloat(totalweightsize, false, true, ParameterSetting.initAdaGrad);
                biasAdaGrad = new CudaPieceFloat(totalbiassize, false, true, ParameterSetting.initAdaGrad);
                if (neuralLinkModel.Nt == N_Type.bLSTM)
                    reweightAdaGrad = new CudaPieceFloat(totalreweightsize, false, true, ParameterSetting.initAdaGrad);
            }
        }

        ~NeuralLinkData()
        {
            Dispose();
        }


        public void Dispose()
        {
            if (layerPoolingOutputs != null)
            {
                foreach (CudaPieceFloat e in layerPoolingOutputs)
                    e.Dispose();
                layerPoolingOutputs = null;
            }
            if (layerMaxPooling_Indices != null)
            {
                foreach (CudaPieceInt e in layerMaxPooling_Indices)
                    e.Dispose();
                layerMaxPooling_Indices = null;
            }

            if (composite_outputs != null)
            {
                foreach (CudaPieceFloat e in composite_outputs)
                    e.Dispose();
                composite_outputs = null;
            }
            if (composite_errors != null)
            {
                foreach (CudaPieceFloat e in composite_errors)
                    e.Dispose();
                composite_errors = null;
            }

            if (weightDeriv != null)
            {
                weightDeriv.Dispose();
            }
            if (biasDeriv != null)
            {
                biasDeriv.Dispose();
            }
            if (reweightDeriv != null)
            {
                reweightDeriv.Dispose();
            }
            if (weightUpdate != null)
            {
                weightUpdate.Dispose();
            }
            if (biasUpdate != null)
            {
                biasUpdate.Dispose();
            }
            if (reweightUpdate != null)
                reweightUpdate.Dispose();
            if (weightAdaGrad != null)
                weightAdaGrad.Dispose();
            if (biasAdaGrad != null)
                biasAdaGrad.Dispose();
            if (reweightAdaGrad != null)
                reweightAdaGrad.Dispose();
            if (layerPoolingInternalState != null)
            {
                foreach (CudaPieceFloat e in layerPoolingInternalState)
                    e.Dispose();
                layerPoolingInternalState = null;
            }
            if (layerPoolingA != null)
            {
                foreach (CudaPieceFloat e in layerPoolingA)
                    e.Dispose();
                layerPoolingA = null;
            }
            if (layerPoolingI != null)
            {
                foreach (CudaPieceFloat e in layerPoolingI)
                    e.Dispose();
                layerPoolingI = null;
            }
            if (layerPoolingF != null)
            {
                foreach (CudaPieceFloat e in layerPoolingF)
                    e.Dispose();
                layerPoolingF = null;
            }
            if (layerPoolingO != null)
            {
                foreach (CudaPieceFloat e in layerPoolingO)
                    e.Dispose();
                layerPoolingO = null;
            }
        }        

        public void ZeroDeriv()
        {
            weightDeriv.Zero();
            biasDeriv.Zero();
            if (neuralLinkModel.Nt == N_Type.bLSTM)
                reweightDeriv.Zero();
        }
    }

    /// <summary>
    /// A particular run related data, including feedforward outputs and backward propagation derivatives
    /// </summary>
    public class DNNRun
    {
        public DNN DnnModel = null;
        public List<NeuralLayerData> neurallayers = new List<NeuralLayerData>();
        public List<NeuralLinkData> neurallinks = new List<NeuralLinkData>();
        public LookupTabRunData wordLT, contextLT;
        Stopwatch sw = new Stopwatch();


        public DNNRun(DNN model)
        {
            DnnModel = model;
            for(int i = 0; i < DnnModel.neurallayers.Count; ++i)
            {
                neurallayers.Add(new NeuralLayerData(DnnModel.neurallayers[i], i != 0));
            }

            for(int i = 0; i < DnnModel.neurallinks.Count; ++i)
            {
                neurallinks.Add(new NeuralLinkData(DnnModel.neurallinks[i]));
            }
            //construct run data for lookup tables
            wordLT = new LookupTabRunData(model.wordLT, true);
            contextLT = new LookupTabRunData(model.contextLT, false);
        }

        public int OutputLayerSize
        {
            get { return neurallayers.Last().Number; }
        }

        /// <summary>
        /// given batch of input data. calculate the output.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="q">indicate which sentence in an instance, 0 -- s1, 1 -- s2, 2 -- s3</param>
        //unsafe public void forward_activate( BatchSample_Input data, List<Amplib.AMPArrayInternal> layerOutputs)
        unsafe public void forward_activate(BatchSample_Input data, int q)
        {
            int layerIndex = 0;
            foreach (NeuralLinkData neurallinkData in neurallinks)
            {
                NeuralLink neurallink = neurallinkData.NeuralLinkModel;
                ///first layer.
                if (layerIndex == 0)
                {
                    if (neurallink.Nt == N_Type.Fully_Connected)
                    {
                        throw new Exception("Not implemented! The first layer must be convolutional or multi-convolutional or birectional LSTM!");
                        //MathOperatorManager.GlobalInstance.SEQ_Sparse_Matrix_Multiply_INTEX(data, neurallink.weight, neurallayers[layerIndex + 1].Outputs[q],
                        //               neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.N_Winsize);
                    }
                    else if (neurallink.Nt == N_Type.Convolution_layer)
                    {
                        MathOperatorManager.GlobalInstance.Convolution_Matrix_Multiply_INTEX(data, neurallink.weight, neurallinkData.LayerPoolingOutputs[q], wordLT.Table, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.N_Winsize);

                        MathOperatorManager.GlobalInstance.Max_Pooling(neurallinkData.LayerPoolingOutputs[q], data, neurallayers[layerIndex + 1].Outputs[q], neurallinkData.LayerMaxPooling_Indices[q], neurallink.Neural_Out.Number, neurallink.N_Winsize);
                    }
                    else if (neurallink.Nt == N_Type.bLSTM)
                    {
                        MathOperatorManager.GlobalInstance.LSTM_Input_Batch_Product(data, neurallink.weight, neurallinkData.LayerPoolingA[q], neurallinkData.LayerPoolingI[q], neurallinkData.LayerPoolingF[q], neurallinkData.LayerPoolingO[q], wordLT.Table, neurallink.Neural_In.Number, neurallink.Neural_Out.Number);

                        MathOperatorManager.GlobalInstance.LSTM_Sequence_Forward(data, neurallink.recur_weight, neurallink.bias, neurallinkData.LayerPoolingA[q], neurallinkData.LayerPoolingI[q], neurallinkData.LayerPoolingF[q], neurallinkData.LayerPoolingO[q], neurallinkData.LayerPoolingInternalState[q], neurallinkData.LayerPoolingOutputs[q], neurallink.Neural_Out.Number);
                        //neurallinkData.LayerPoolingA[q].CopyOutFromCuda();
                        //neurallinkData.LayerPoolingI[q].CopyOutFromCuda();
                        //neurallinkData.LayerPoolingF[q].CopyOutFromCuda();
                        //neurallinkData.LayerPoolingO[q].CopyOutFromCuda();
                        //neurallinkData.LayerPoolingInternalState[q].CopyOutFromCuda();
                        //neurallinkData.LayerPoolingOutputs[q].CopyOutFromCuda();

                        MathOperatorManager.GlobalInstance.LSTM_Max_Pooling(neurallinkData.LayerPoolingOutputs[q], data, neurallayers[layerIndex + 1].Outputs[q], neurallinkData.LayerMaxPooling_Indices[q], neurallink.Neural_Out.Number);
                    }
                    else  // must be multi-convolutional, composite layer cannot be the first layer
                    {
                        MathOperatorManager.GlobalInstance.MultiConv_Matrix_Multiply_INTEX(data, neurallink.weight, neurallinkData.LayerPoolingOutputs[q], wordLT.Table, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.winsizes, neurallink.fmsizes);

                        //CudaPieceInt test1 = new CudaPieceInt(1, true, true);
                        //test1.MemPtr[0] = 1;
                        //test1.CopyIntoCuda();
                        //CudaPieceInt test2 = new CudaPieceInt(1, true, true);
                        //Program.Print("Success!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

                        MathOperatorManager.GlobalInstance.Multi_Max_Pooling(neurallinkData.LayerPoolingOutputs[q], data, neurallayers[layerIndex + 1].Outputs[q], neurallinkData.LayerMaxPooling_Indices[q], neurallink.Neural_Out.Number, neurallink.winsizes, neurallink.fmsizes);
                    }
                }
                else
                {
                    if (neurallink.Nt == N_Type.Composite_Full)
                    {
                        MathOperatorManager.GlobalInstance.FillOut_Composite(neurallayers[layerIndex].Outputs[q], data, neurallinkData.CompOutputs[q], contextLT.Table, contextLT.InputDeriv[q], neurallink.Neural_In.Number, neurallink.Extra_Input.Number, 1);
                        MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallinkData.CompOutputs[q], neurallink.weight, neurallayers[layerIndex + 1].Outputs[q], data.batchsize, neurallink.Neural_In.Number + neurallink.Extra_Input.Number, neurallink.Neural_Out.Number, 0);
                    }
                    else
                        MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallayers[layerIndex].Outputs[q], neurallink.weight, neurallayers[layerIndex + 1].Outputs[q], data.batchsize, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, 0);
                }


                if (neurallink.Nt != N_Type.bLSTM)
                {
                    if (neurallink.Af == A_Func.Tanh)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Add_Tanh(neurallayers[layerIndex + 1].Outputs[q], neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                    }
                    else if (neurallink.Af == A_Func.Linear)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Add_Vector(neurallayers[layerIndex + 1].Outputs[q], neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                    }
                    else if (neurallink.Af == A_Func.Rectified)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Rectified_Vector(neurallayers[layerIndex + 1].Outputs[q], neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                    }
                }

                layerIndex += 1;
            }
        }

        /// <summary>
        /// BackProp the error derivative on the output of each layer.
        /// The output layer's errorDeriv must be previuosly setup.
        /// </summary>
        unsafe public void backward_calculate_layerout_deriv(BatchSample_Input input_batch, int q)
        {
            int batchsize = input_batch.batchsize;
            for (int i = neurallinks.Count - 1; i > 0; i--)
            {
                NeuralLink nlink = neurallinks[i].NeuralLinkModel;
                if (nlink.Nt == N_Type.Fully_Connected)
                {
                    MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallayers[i + 1].ErrorDerivs[q], nlink.weight, neurallayers[i].ErrorDerivs[q], batchsize,
                            nlink.Neural_Out.Number, nlink.Neural_In.Number, 1);
                }
                else // must be composite full
                {
                    MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallayers[i + 1].ErrorDerivs[q], nlink.weight, neurallinks[i].CompErrors[q], batchsize,
                            nlink.Neural_Out.Number, (nlink.Neural_In.Number*nlink.N_Winsize + nlink.Extra_Input.Number), 1);
                    MathOperatorManager.GlobalInstance.FillOut_Composite(neurallayers[i].ErrorDerivs[q], input_batch, neurallinks[i].CompErrors[q], contextLT.Table, contextLT.InputDeriv[q],
                            nlink.Neural_In.Number, nlink.Extra_Input.Number, 0);
                }

                if (i != 1 || neurallinks[0].NeuralLinkModel.Nt != N_Type.bLSTM)
                {
                    if (neurallinks[i - 1].NeuralLinkModel.Af == A_Func.Tanh)
                    {
                        MathOperatorManager.GlobalInstance.Deriv_Tanh(neurallayers[i].ErrorDerivs[q], neurallayers[i].Outputs[q], batchsize, nlink.Neural_In.Number);
                    }
                    else if (neurallinks[i - 1].NeuralLinkModel.Af == A_Func.Rectified)
                    {
                        MathOperatorManager.GlobalInstance.Deriv_Rectified(neurallayers[i].ErrorDerivs[q], neurallayers[i].Outputs[q], batchsize, nlink.Neural_In.Number);
                    }
                }
            }

            if (neurallinks[0].NeuralLinkModel.Nt == N_Type.bLSTM)
            {
                // error back propagation in time
                MathOperatorManager.GlobalInstance.LSTM_Sequence_Backward(input_batch, neurallinks[0].NeuralLinkModel.recur_weight,
                    neurallinks[0].LayerMaxPooling_Indices[q], neurallayers[1].ErrorDerivs[q], neurallinks[0].LayerPoolingA[q],
                    neurallinks[0].LayerPoolingI[q], neurallinks[0].LayerPoolingF[q], neurallinks[0].LayerPoolingO[q],
                    neurallinks[0].LayerPoolingInternalState[q], neurallinks[0].LayerPoolingOutputs[q], neurallinks[0].NeuralLinkModel.Neural_Out.Number);

                neurallinks[0].LayerPoolingA[q].CopyOutFromCuda();
                neurallinks[0].LayerPoolingI[q].CopyOutFromCuda();
                neurallinks[0].LayerPoolingF[q].CopyOutFromCuda();
                neurallinks[0].LayerPoolingO[q].CopyOutFromCuda();
            }
        }

        unsafe public void backward_calculate_weight_deriv(BatchSample_Input[] input_batches) // 0-q1,1-q2,2-q3
        {
            //Calculate derivatives for all parameters, including link weight, bias and lookup tables. The derivatives for context table have already been computed 
            int batchsize = input_batches[0].batchsize;
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].ZeroDeriv();

                if (ParameterSetting.UpdateBias && neurallinks[i].NeuralLinkModel.Nt != N_Type.bLSTM)
                {
                    MathOperatorManager.GlobalInstance.Matrix_Aggragate(neurallayers[i + 1].ErrorDerivs[0], neurallayers[i + 1].ErrorDerivs[1], neurallayers[i + 1].ErrorDerivs[2], neurallinks[i].BiasDeriv, batchsize, neurallinks[i].NeuralLinkModel.Neural_Out.Number);
                }

                if (i == 0)
                {
                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.Fully_Connected)
                    {
                        throw new Exception("Not implemented! The first layer must be convolutional or multi-convolutional or bidirectional LSTM!");
                        //MathOperatorManager.GlobalInstance.SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(input_batch, neurallinks[i].WeightDeriv, neurallayers[i + 1].ErrorDeriv, neurallinks[i].NeuralLinkModel.Neural_In.Number, neurallinks[i].NeuralLinkModel.Neural_Out.Number, neurallinks[i].NeuralLinkModel.N_Winsize);
                    }
                    else if (neurallinks[i].NeuralLinkModel.Nt == N_Type.Convolution_layer)
                    {
                        MathOperatorManager.GlobalInstance.Convolution_Matrix_Product_INTEX(neurallayers[i + 1].ErrorDerivs[0], neurallinks[i].LayerMaxPooling_Indices[0], neurallayers[i + 1].ErrorDerivs[1], neurallinks[i].LayerMaxPooling_Indices[1], neurallayers[i + 1].ErrorDerivs[2], neurallinks[i].LayerMaxPooling_Indices[2], wordLT.Table, 
                                     input_batches[0], input_batches[1], input_batches[2], neurallinks[i].NeuralLinkModel.N_Winsize,
                                     batchsize, neurallayers[i + 1].Number, neurallinks[i].WeightDeriv, neurallinks[i].NeuralLinkModel.Neural_In.Number);
                    }
                    else if (neurallinks[i].NeuralLinkModel.Nt == N_Type.bLSTM)
                    {
                        MathOperatorManager.GlobalInstance.LSTM_Weight_Deriv(input_batches[0], input_batches[1], input_batches[2], wordLT.Table, neurallinks[i].WeightDeriv,
                            neurallinks[i].LayerPoolingA[0], neurallinks[i].LayerPoolingA[1], neurallinks[i].LayerPoolingA[2], neurallinks[i].LayerPoolingI[0],
                            neurallinks[i].LayerPoolingI[1], neurallinks[i].LayerPoolingI[2], neurallinks[i].LayerPoolingF[0], neurallinks[i].LayerPoolingF[1],
                            neurallinks[i].LayerPoolingF[2], neurallinks[i].LayerPoolingO[0], neurallinks[i].LayerPoolingO[1], neurallinks[i].LayerPoolingO[2],
                            neurallinks[i].LayerPoolingOutputs[0], neurallinks[i].LayerPoolingOutputs[1], neurallinks[i].LayerPoolingOutputs[2], neurallinks[i].NeuralLinkModel.Neural_In.Number,
                            neurallinks[i].NeuralLinkModel.Neural_Out.Number, 0);
                        MathOperatorManager.GlobalInstance.LSTM_Weight_Deriv(input_batches[0], input_batches[1], input_batches[2], wordLT.Table, neurallinks[i].reWeightDeriv,
                            neurallinks[i].LayerPoolingA[0], neurallinks[i].LayerPoolingA[1], neurallinks[i].LayerPoolingA[2], neurallinks[i].LayerPoolingI[0],
                            neurallinks[i].LayerPoolingI[1], neurallinks[i].LayerPoolingI[2], neurallinks[i].LayerPoolingF[0], neurallinks[i].LayerPoolingF[1],
                            neurallinks[i].LayerPoolingF[2], neurallinks[i].LayerPoolingO[0], neurallinks[i].LayerPoolingO[1], neurallinks[i].LayerPoolingO[2],
                            neurallinks[i].LayerPoolingOutputs[0], neurallinks[i].LayerPoolingOutputs[1], neurallinks[i].LayerPoolingOutputs[2], neurallinks[i].NeuralLinkModel.Neural_In.Number,
                            neurallinks[i].NeuralLinkModel.Neural_Out.Number, 1);
                        MathOperatorManager.GlobalInstance.LSTM_Bias_Deriv(input_batches[0], input_batches[1], input_batches[2], neurallinks[i].BiasDeriv,
                            neurallinks[i].LayerPoolingA[0], neurallinks[i].LayerPoolingA[1], neurallinks[i].LayerPoolingA[2], neurallinks[i].LayerPoolingI[0],
                            neurallinks[i].LayerPoolingI[1], neurallinks[i].LayerPoolingI[2], neurallinks[i].LayerPoolingF[0], neurallinks[i].LayerPoolingF[1],
                            neurallinks[i].LayerPoolingF[2], neurallinks[i].LayerPoolingO[0], neurallinks[i].LayerPoolingO[1], neurallinks[i].LayerPoolingO[2],
                            neurallinks[i].NeuralLinkModel.Neural_Out.Number);
                    }
                    else // must be multi-convolutional, composite layer cannot be the first layer
                    {
                        int accu = 0;
                        for (int b = 0; b < neurallinks[i].NeuralLinkModel.num_fms.Length; b++)
                        {
                            MathOperatorManager.GlobalInstance.MultiConv_Matrix_Product_INTEX(neurallayers[i + 1].ErrorDerivs[0], neurallinks[i].LayerMaxPooling_Indices[0], neurallayers[i + 1].ErrorDerivs[1], neurallinks[i].LayerMaxPooling_Indices[1], neurallayers[i + 1].ErrorDerivs[2], neurallinks[i].LayerMaxPooling_Indices[2], wordLT.Table,
                                     input_batches[0], input_batches[1], input_batches[2],
                                     batchsize, neurallayers[i + 1].Number, neurallinks[i].WeightDeriv, neurallinks[i].NeuralLinkModel.Neural_In.Number, neurallinks[i].NeuralLinkModel.wnd_sizes[b], neurallinks[i].NeuralLinkModel.num_fms[b], accu, b>0?neurallinks[i].NeuralLinkModel.ma_sizes[b-1]:0);
                            accu += neurallinks[i].NeuralLinkModel.num_fms[b];
                        }                     
                    }
                }
                else
                {
                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.Fully_Connected)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Product(neurallayers[i].Outputs[0], neurallayers[i + 1].ErrorDerivs[0], neurallayers[i].Outputs[1], neurallayers[i + 1].ErrorDerivs[1], neurallayers[i].Outputs[2], neurallayers[i + 1].ErrorDerivs[2], neurallinks[i].WeightDeriv,
                                                batchsize, neurallayers[i].Number, neurallayers[i + 1].Number);
                    }
                    else // must be composite full
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Product(neurallinks[i].CompOutputs[0], neurallayers[i + 1].ErrorDerivs[0], neurallinks[i].CompOutputs[1], neurallayers[i + 1].ErrorDerivs[1], neurallinks[i].CompOutputs[2], neurallayers[i + 1].ErrorDerivs[2], neurallinks[i].WeightDeriv,
                                                batchsize, (neurallayers[i].Number + neurallinks[i].NeuralLinkModel.Extra_Input.Number), neurallayers[i + 1].Number);
                    }
                    
                }
            }
            //Finally, word vector derivatives
            wordLT.ZeroDeriv();
            if (neurallinks[0].NeuralLinkModel.Nt == N_Type.MultiWidthConv_layer)
            {
                for (int i = 0; i < 3; i++)
                {
                    MathOperatorManager.GlobalInstance.MultiConv_Compute_WVDERIV(neurallayers[1].ErrorDerivs[i], neurallinks[0].LayerMaxPooling_Indices[i], neurallinks[0].NeuralLinkModel.weight, batchsize,
                                neurallayers[1].Number, wordLT.InputDeriv[i], neurallinks[0].NeuralLinkModel.Neural_In.Number, neurallinks[0].NeuralLinkModel.winsizes, neurallinks[0].NeuralLinkModel.fmsizes);
                }
            }
            else if (neurallinks[0].NeuralLinkModel.Nt == N_Type.bLSTM)
            {
                for (int i = 0; i < 3; i++)
                {
                    MathOperatorManager.GlobalInstance.LSTM_Compute_WVDeriv(input_batches[i].elementSize, neurallinks[0].NeuralLinkModel.weight, wordLT.InputDeriv[i], neurallinks[0].LayerPoolingA[i],
                        neurallinks[0].LayerPoolingI[i], neurallinks[0].LayerPoolingF[i], neurallinks[0].LayerPoolingO[i], neurallinks[0].NeuralLinkModel.Neural_In.Number, neurallinks[0].NeuralLinkModel.Neural_Out.Number);
                }
            }
            else // convolution
            {
                for (int i = 0; i < 3; i++)
                {
                    MathOperatorManager.GlobalInstance.Conv_Compute_WVDERIV(neurallayers[1].ErrorDerivs[i], neurallinks[0].LayerMaxPooling_Indices[i], neurallinks[0].NeuralLinkModel.weight, batchsize,
                                neurallayers[1].Number, wordLT.InputDeriv[i], neurallinks[0].NeuralLinkModel.Neural_In.Number, neurallinks[0].NeuralLinkModel.N_Winsize);
                }
            }

            if (ParameterSetting.CheckGrad)
            {
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Update(wordLT.TabUpdate, wordLT, input_batches[0].elementSize, input_batches[1].elementSize, wordLT.Table.vecDim, 1.0f);
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Update(contextLT.TabUpdate, contextLT, input_batches[0].batchsize, input_batches[1].batchsize, contextLT.Table.vecDim, 1.0f);
            }
        }
                
        /// <summary>
        /// the error deriv at top output layer must have been set up before call this method.
        /// This process only do backprop computations. It does not update model weights at all.
        /// Need to call update_weight afterwards to update models.
        /// </summary>
        /// <param name="input_batch"></param>
        /// <param name="momentum"></param>
        /// <param name="learning_rate"></param>
        public void backward_propagate_deriv(BatchSample_Input[] input_batches)
        {
            // step 1, compute the derivatives for the output values of each layer
            for (int q = 0; q < 3; q++)
                backward_calculate_layerout_deriv(input_batches[q], q);
            // step 2, compute the derivatives for the connections of each neural link layer
            summarizeUnique(input_batches);
            backward_calculate_weight_deriv(input_batches);
        }

        /// <summary>
        /// Must call backward_propagate(), or two steps one by one, before calling this method.
        /// </summary>
        unsafe public void update_weight(BatchSample_Input[] input_batches, float momentum, float learning_rate)
        {
            /// First, update weights and bias
            /// step 1, compute the weight updates, taking momentum and learning rates into consideration
            /// Wei_Update = momentum * Wei_Update + learn_rate * grad.
            int row = 0, col = 0, rerow = 0, recol = 0;
            
            for (int i = 0; i < neurallinks.Count; i++)
            {
                int biassize = neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                if (neurallinks[i].NeuralLinkModel.Nt == N_Type.Composite_Full)
                {
                    row = neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                    col = neurallinks[i].NeuralLinkModel.Neural_In.Number * neurallinks[i].NeuralLinkModel.N_Winsize + neurallinks[i].NeuralLinkModel.Extra_Input.Number;
                }
                else if (neurallinks[i].NeuralLinkModel.Nt == N_Type.MultiWidthConv_layer)
                {
                    // treat feature dimensin as num of rows
                    row = neurallinks[i].NeuralLinkModel.Neural_In.Number;
                    col = neurallinks[i].NeuralLinkModel.ma_sizes[neurallinks[i].NeuralLinkModel.ma_sizes.Length - 1] / row;
                }
                else if (neurallinks[i].NeuralLinkModel.Nt == N_Type.bLSTM)
                {
                    row = neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                    col = neurallinks[i].NeuralLinkModel.Neural_In.Number * 4;
                    rerow = neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                    recol = neurallinks[i].NeuralLinkModel.Neural_Out.Number * 2;
                    biassize = neurallinks[i].NeuralLinkModel.Neural_Out.Number * 4;
                }
                else
                {
                    row = neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                    col = neurallinks[i].NeuralLinkModel.Neural_In.Number * neurallinks[i].NeuralLinkModel.N_Winsize;
                }

                if (ParameterSetting.updateScheme == 1)
                {
                    // add the momentum
                    MathOperatorManager.GlobalInstance.Scale_Matrix(neurallinks[i].WeightUpdate, col, row, momentum);

                    // dnn_neurallinks[i].Weight
                    MathOperatorManager.GlobalInstance.Matrix_Add(neurallinks[i].WeightUpdate, neurallinks[i].WeightDeriv, col, row, learning_rate);

                    // update the model: Weight = Weight += Wei_Update
                    MathOperatorManager.GlobalInstance.Matrix_Add_REAL(neurallinks[i].NeuralLinkModel.weight, neurallinks[i].WeightUpdate, col, row);

                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.bLSTM)
                    {
                        MathOperatorManager.GlobalInstance.Scale_Matrix(neurallinks[i].reWeightUpdate, recol, rerow, momentum);
                        MathOperatorManager.GlobalInstance.Matrix_Add(neurallinks[i].reWeightUpdate, neurallinks[i].reWeightDeriv, recol, rerow, learning_rate);
                        MathOperatorManager.GlobalInstance.Matrix_Add_REAL(neurallinks[i].NeuralLinkModel.recur_weight, neurallinks[i].reWeightUpdate, recol, rerow);
                    }
                }
                else if (ParameterSetting.updateScheme == 0)
                {
                    MathOperatorManager.GlobalInstance.Matrix_Grad_Decent(neurallinks[i].NeuralLinkModel.weight, neurallinks[i].WeightDeriv, col, row, learning_rate);
                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.bLSTM)
                        MathOperatorManager.GlobalInstance.Matrix_Grad_Decent(neurallinks[i].NeuralLinkModel.recur_weight, neurallinks[i].reWeightDeriv, recol, rerow, learning_rate);
                }
                else // AdaGrad
                {
                    MathOperatorManager.GlobalInstance.Matrix_Ada_Grad_Decent(neurallinks[i].NeuralLinkModel.weight, neurallinks[i].WeightDeriv, neurallinks[i].WeightAdaGrad, col, row, learning_rate, ParameterSetting.DSSMEpsilon);
                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.bLSTM)
                        MathOperatorManager.GlobalInstance.Matrix_Ada_Grad_Decent(neurallinks[i].NeuralLinkModel.recur_weight, neurallinks[i].reWeightDeriv, neurallinks[i].reWeightAdaGrad, recol, rerow, learning_rate, ParameterSetting.DSSMEpsilon);
                }

                if (ParameterSetting.UpdateBias)
                {
                    if (ParameterSetting.updateScheme == 1)
                    {
                        // add the momentum
                        MathOperatorManager.GlobalInstance.Scale_Matrix(neurallinks[i].BiasUpdate, 1, biassize, momentum);

                        // dnn_neurallinks[i].Weight
                        MathOperatorManager.GlobalInstance.Matrix_Add(neurallinks[i].BiasUpdate, neurallinks[i].BiasDeriv, 1, biassize, learning_rate);
                        // upate the model
                        MathOperatorManager.GlobalInstance.Matrix_Add_REAL(neurallinks[i].NeuralLinkModel.bias, neurallinks[i].BiasUpdate, 1, biassize);
                    }
                    else if (ParameterSetting.updateScheme == 0)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Grad_Decent(neurallinks[i].NeuralLinkModel.bias, neurallinks[i].BiasDeriv, 1, biassize, learning_rate);
                    }
                    else // AdaGrad
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Ada_Grad_Decent(neurallinks[i].NeuralLinkModel.bias, neurallinks[i].BiasDeriv, neurallinks[i].BiasAdaGrad, 1, biassize, learning_rate, ParameterSetting.DSSMEpsilon);
                    }
                }
            }

            // update lookup tables
            if (ParameterSetting.updateScheme == 0)
            {
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup(wordLT.Table, wordLT, input_batches[0].elementSize, input_batches[1].elementSize, wordLT.Table.vecDim, learning_rate);
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup(contextLT.Table, contextLT, input_batches[0].batchsize, input_batches[1].batchsize, contextLT.Table.vecDim, learning_rate);
            }
            else if (ParameterSetting.updateScheme == 1)
            {
                MathOperatorManager.GlobalInstance.Scale_Matrix(wordLT.TabUpdate, wordLT.Table.count, wordLT.Table.vecDim, momentum);
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Update(wordLT.TabUpdate, wordLT, input_batches[0].elementSize, input_batches[1].elementSize, wordLT.Table.vecDim, learning_rate);
                MathOperatorManager.GlobalInstance.Matrix_Add_REAL(wordLT.Table.table, wordLT.TabUpdate, wordLT.Table.count, wordLT.Table.vecDim);

                MathOperatorManager.GlobalInstance.Scale_Matrix(contextLT.TabUpdate, contextLT.Table.count, contextLT.Table.vecDim, momentum);
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Update(contextLT.TabUpdate, contextLT, input_batches[0].batchsize, input_batches[1].batchsize, contextLT.Table.vecDim, learning_rate);
                MathOperatorManager.GlobalInstance.Matrix_Add_REAL(contextLT.Table.table, contextLT.TabUpdate, contextLT.Table.count, contextLT.Table.vecDim);
            }
            else // AdaGrad
            {
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Ada(wordLT.Table, wordLT, input_batches[0].elementSize, input_batches[1].elementSize, wordLT.Table.vecDim, learning_rate, ParameterSetting.DSSMEpsilon);
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Ada(contextLT.Table, contextLT, input_batches[0].batchsize, input_batches[1].batchsize, contextLT.Table.vecDim, learning_rate, ParameterSetting.DSSMEpsilon);
            }

        }

        unsafe void summarizeUnique(BatchSample_Input[] inputBatches)
        {
            if (ParameterSetting.DEBUG)
            {
                sw.Reset();
                sw.Start();
            }
            Dictionary<int, List<int>> wordSum = new Dictionary<int, List<int>>(10000);
            int offset = 0;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < inputBatches[i].elementSize; j++)
                {
                    int w = inputBatches[i].Word_Idx_Mem[j];
                    if (wordSum.ContainsKey(w))
                        wordSum[w].Add(offset);
                    else
                    {
                        wordSum.Add(w, new List<int>());
                        wordSum[w].Add(offset);
                    }
                    offset++;
                }
            }
            int c1 = 0, c2 = 0;
            foreach (KeyValuePair<int, List<int>> d in wordSum)
            {
                wordLT.uniqueWordID.MemPtr[c1] = d.Key;
                for (int i = 0; i < d.Value.Count; i++)
                {
                    wordLT.Sequence.MemPtr[c2] = d.Value[i];
                    c2++;
                }
                wordLT.uniqueWordIdx.MemPtr[c1] = c2;
                c1++;
            }
            wordLT.uniqueNum = c1;

            wordSum.Clear();
            offset = 0;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < inputBatches[i].batchsize; j++)
                {
                    int w = inputBatches[i].Fea_Mem[j];
                    if (wordSum.ContainsKey(w))
                        wordSum[w].Add(offset);
                    else
                    {
                        wordSum.Add(w, new List<int>());
                        wordSum[w].Add(offset);
                    }
                    offset++;
                }
            }
            c1 = 0; 
            c2 = 0;
            foreach (KeyValuePair<int, List<int>> d in wordSum)
            {
                contextLT.uniqueWordID.MemPtr[c1] = d.Key;
                for (int i = 0; i < d.Value.Count; i++)
                {
                    contextLT.Sequence.MemPtr[c2] = d.Value[i];
                    c2++;
                }
                contextLT.uniqueWordIdx.MemPtr[c1] = c2;
                c1++;
            }
            contextLT.uniqueNum = c1;

            wordLT.uniqueWordID.CopyIntoCuda();
            wordLT.uniqueWordIdx.CopyIntoCuda();
            wordLT.Sequence.CopyIntoCuda();

            contextLT.uniqueWordID.CopyIntoCuda();
            contextLT.uniqueWordIdx.CopyIntoCuda();
            contextLT.Sequence.CopyIntoCuda();

            if (ParameterSetting.DEBUG)
            {
                sw.Stop();
                Program.Print("==================>SummarizeUnique time used: " + sw.Elapsed.ToString());
            }
        }
    }

    public class DNNRunForward
    {
        public DNN DnnModel = null;
        public List<NeuralLayerDataSup> neurallayers = new List<NeuralLayerDataSup>();
        public List<NeuralLinkDataSup> neurallinks = new List<NeuralLinkDataSup>();
        
        //Stopwatch sw = new Stopwatch();


        public DNNRunForward(DNN model, int maxBatchsize, int maxSegsize)
        {
            DnnModel = model;
            for (int i = 0; i < DnnModel.neurallayers.Count; ++i)
            {
                neurallayers.Add(new NeuralLayerDataSup(DnnModel.neurallayers[i], maxBatchsize, i != 0, true, i == (DnnModel.neurallayers.Count-1)));
            }

            for (int i = 0; i < DnnModel.neurallinks.Count; ++i)
            {
                neurallinks.Add(new NeuralLinkDataSup(DnnModel.neurallinks[i], maxBatchsize, maxSegsize, true));
            }
            
        }

        public void Dispose()
        {
            if (neurallayers != null)
            {
                for (int i = 0; i < neurallayers.Count; i++)
                    neurallayers[i].Dispose();
                neurallayers.Clear();
            }
            if (neurallinks != null)
            {
                for (int i = 0; i < neurallinks.Count; i++)
                    neurallinks[i].Dispose();
                neurallinks.Clear();
            }
        }

        public int OutputLayerSize
        {
            get { return neurallayers.Last().Number; }
        }

        /// <summary>
        /// given batch of input data. calculate the output.
        /// </summary>
        /// <param name="data"></param>
        //unsafe public void forward_activate( BatchSample_Input data, List<Amplib.AMPArrayInternal> layerOutputs)
        unsafe public void forward_activate(BatchSample_Input data)
        {
            int layerIndex = 0;
            foreach (NeuralLinkDataSup neurallinkData in neurallinks)
            {
                NeuralLink neurallink = neurallinkData.NeuralLinkModel;
                ///first layer.
                if (layerIndex == 0)
                {
                    if (neurallink.Nt == N_Type.Fully_Connected)
                    {
                        throw new Exception("Not implemented! The first layer must be convolutional or multi-convolutional!");
                        //MathOperatorManager.GlobalInstance.SEQ_Sparse_Matrix_Multiply_INTEX(data, neurallink.weight, neurallayers[layerIndex + 1].Outputs[q],
                        //               neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.N_Winsize);
                    }
                    else if (neurallink.Nt == N_Type.Convolution_layer)
                    {
                        MathOperatorManager.GlobalInstance.Convolution_Matrix_Multiply_INTEX(data, neurallink.weight, neurallinkData.LayerPoolingOutput, DnnModel.wordLT, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.N_Winsize);

                        MathOperatorManager.GlobalInstance.Max_Pooling(neurallinkData.LayerPoolingOutput, data, neurallayers[layerIndex + 1].Output, neurallinkData.LayerMaxPooling_Index, neurallink.Neural_Out.Number, neurallink.N_Winsize);
                    }
                    else if (neurallink.Nt == N_Type.bLSTM)
                    {
                        MathOperatorManager.GlobalInstance.LSTM_Input_Batch_Product(data, neurallink.weight, neurallinkData.LayerPoolingA, neurallinkData.LayerPoolingI, neurallinkData.LayerPoolingF, neurallinkData.LayerPoolingO, DnnModel.wordLT, neurallink.Neural_In.Number, neurallink.Neural_Out.Number);
                        MathOperatorManager.GlobalInstance.LSTM_Sequence_Forward(data, neurallink.recur_weight, neurallink.bias, neurallinkData.LayerPoolingA, neurallinkData.LayerPoolingI, neurallinkData.LayerPoolingF, neurallinkData.LayerPoolingO, neurallinkData.LayerPoolingInternalState, neurallinkData.LayerPoolingOutput, neurallink.Neural_Out.Number);
                        MathOperatorManager.GlobalInstance.LSTM_Max_Pooling(neurallinkData.LayerPoolingOutput, data, neurallayers[layerIndex + 1].Output, neurallinkData.LayerMaxPooling_Index, neurallink.Neural_Out.Number);
                    }
                    else  // must be multi-convolutional, composite layer cannot be the first layer
                    {
                        MathOperatorManager.GlobalInstance.MultiConv_Matrix_Multiply_INTEX(data, neurallink.weight, neurallinkData.LayerPoolingOutput, DnnModel.wordLT, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.winsizes, neurallink.fmsizes);

                        MathOperatorManager.GlobalInstance.Multi_Max_Pooling(neurallinkData.LayerPoolingOutput, data, neurallayers[layerIndex + 1].Output, neurallinkData.LayerMaxPooling_Index, neurallink.Neural_Out.Number, neurallink.winsizes, neurallink.fmsizes);
                    }
                }
                else
                {
                    if (neurallink.Nt == N_Type.Composite_Full)
                    {
                        MathOperatorManager.GlobalInstance.FillOut_Composite(neurallayers[layerIndex].Output, data, neurallinkData.CompOutput, DnnModel.contextLT, null, neurallink.Neural_In.Number, neurallink.Extra_Input.Number, 1);
                        MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallinkData.CompOutput, neurallink.weight, neurallayers[layerIndex + 1].Output, data.batchsize, neurallink.Neural_In.Number + neurallink.Extra_Input.Number, neurallink.Neural_Out.Number, 0);
                    }
                    else
                        MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallayers[layerIndex].Output, neurallink.weight, neurallayers[layerIndex + 1].Output, data.batchsize, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, 0);
                }

                if (neurallink.Nt != N_Type.bLSTM)
                {
                    if (neurallink.Af == A_Func.Tanh)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Add_Tanh(neurallayers[layerIndex + 1].Output, neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                    }
                    else if (neurallink.Af == A_Func.Linear)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Add_Vector(neurallayers[layerIndex + 1].Output, neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                    }
                    else if (neurallink.Af == A_Func.Rectified)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Rectified_Vector(neurallayers[layerIndex + 1].Output, neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                    }
                }
                layerIndex += 1;
            }
        }
    }


    public class DNNRunSup
    {
        public DNN DnnModel = null;
        public List<NeuralLayerDataSup> neurallayers = new List<NeuralLayerDataSup>();
        public List<NeuralLinkDataSup> neurallinks = new List<NeuralLinkDataSup>();
        public LookupTabRunDataSup wordLT, contextLT;
        //Stopwatch sw = new Stopwatch();


        public DNNRunSup(DNN model, int maxBatchsize, int maxSegsize)
        {
            DnnModel = model;
            for (int i = 0; i < DnnModel.neurallayers.Count; ++i)
            {
                neurallayers.Add(new NeuralLayerDataSup(DnnModel.neurallayers[i], maxBatchsize, i != 0, false, i == (DnnModel.neurallayers.Count - 1)));
            }

            for (int i = 0; i < DnnModel.neurallinks.Count; ++i)
            {
                neurallinks.Add(new NeuralLinkDataSup(DnnModel.neurallinks[i], maxBatchsize, maxSegsize, false));
            }
            //construct run data for lookup tables
            wordLT = new LookupTabRunDataSup(model.wordLT, true, maxSegsize, maxBatchsize);
            contextLT = new LookupTabRunDataSup(model.contextLT, false, maxSegsize, maxBatchsize);
        }

        public int OutputLayerSize
        {
            get { return neurallayers.Last().Number; }
        }

        /// <summary>
        /// given batch of input data. calculate the output.
        /// </summary>
        /// <param name="data"></param>
        //unsafe public void forward_activate( BatchSample_Input data, List<Amplib.AMPArrayInternal> layerOutputs)
        unsafe public void forward_activate(BatchSample_Input data)
        {
            int layerIndex = 0;
            foreach (NeuralLinkDataSup neurallinkData in neurallinks)
            {
                NeuralLink neurallink = neurallinkData.NeuralLinkModel;
                ///first layer.
                if (layerIndex == 0)
                {
                    if (neurallink.Nt == N_Type.Fully_Connected)
                    {
                        throw new Exception("Not implemented! The first layer must be convolutional or multi-convolutional!");
                        //MathOperatorManager.GlobalInstance.SEQ_Sparse_Matrix_Multiply_INTEX(data, neurallink.weight, neurallayers[layerIndex + 1].Outputs[q],
                        //               neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.N_Winsize);
                    }
                    else if (neurallink.Nt == N_Type.Convolution_layer)
                    {
                        MathOperatorManager.GlobalInstance.Convolution_Matrix_Multiply_INTEX(data, neurallink.weight, neurallinkData.LayerPoolingOutput, DnnModel.wordLT, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.N_Winsize);

                        MathOperatorManager.GlobalInstance.Max_Pooling(neurallinkData.LayerPoolingOutput, data, neurallayers[layerIndex + 1].Output, neurallinkData.LayerMaxPooling_Index, neurallink.Neural_Out.Number, neurallink.N_Winsize);
                    }
                    else if (neurallink.Nt == N_Type.bLSTM)
                    {
                        MathOperatorManager.GlobalInstance.LSTM_Input_Batch_Product(data, neurallink.weight, neurallinkData.LayerPoolingA, neurallinkData.LayerPoolingI, neurallinkData.LayerPoolingF, neurallinkData.LayerPoolingO, DnnModel.wordLT, neurallink.Neural_In.Number, neurallink.Neural_Out.Number);
                        MathOperatorManager.GlobalInstance.LSTM_Sequence_Forward(data, neurallink.recur_weight, neurallink.bias, neurallinkData.LayerPoolingA, neurallinkData.LayerPoolingI, neurallinkData.LayerPoolingF, neurallinkData.LayerPoolingO, neurallinkData.LayerPoolingInternalState, neurallinkData.LayerPoolingOutput, neurallink.Neural_Out.Number);
                        MathOperatorManager.GlobalInstance.LSTM_Max_Pooling(neurallinkData.LayerPoolingOutput, data, neurallayers[layerIndex + 1].Output, neurallinkData.LayerMaxPooling_Index, neurallink.Neural_Out.Number);
                    }
                    else  // must be multi-convolutional, composite layer cannot be the first layer
                    {
                        MathOperatorManager.GlobalInstance.MultiConv_Matrix_Multiply_INTEX(data, neurallink.weight, neurallinkData.LayerPoolingOutput, DnnModel.wordLT, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.winsizes, neurallink.fmsizes);

                        MathOperatorManager.GlobalInstance.Multi_Max_Pooling(neurallinkData.LayerPoolingOutput, data, neurallayers[layerIndex + 1].Output, neurallinkData.LayerMaxPooling_Index, neurallink.Neural_Out.Number, neurallink.winsizes, neurallink.fmsizes);
                    }
                }
                else
                {
                    if (neurallink.Nt == N_Type.Composite_Full)
                    {
                        MathOperatorManager.GlobalInstance.FillOut_Composite(neurallayers[layerIndex].Output, data, neurallinkData.CompOutput, DnnModel.contextLT, null, neurallink.Neural_In.Number, neurallink.Extra_Input.Number, 1);
                        MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallinkData.CompOutput, neurallink.weight, neurallayers[layerIndex + 1].Output, data.batchsize, neurallink.Neural_In.Number + neurallink.Extra_Input.Number, neurallink.Neural_Out.Number, 0);
                    }
                    else
                        MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallayers[layerIndex].Output, neurallink.weight, neurallayers[layerIndex + 1].Output, data.batchsize, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, 0);
                }

                if (neurallink.Nt != N_Type.bLSTM)
                {
                    if (neurallink.Af == A_Func.Tanh)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Add_Tanh(neurallayers[layerIndex + 1].Output, neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                    }
                    else if (neurallink.Af == A_Func.Linear)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Add_Vector(neurallayers[layerIndex + 1].Output, neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                    }
                    else if (neurallink.Af == A_Func.Rectified)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Rectified_Vector(neurallayers[layerIndex + 1].Output, neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                    }
                }
                layerIndex += 1;
            }
        }

        /// <summary>
        /// BackProp the error derivative on the output of each layer.
        /// The output layer's errorDeriv must be previuosly setup.
        /// </summary>
        unsafe public void backward_calculate_layerout_deriv(BatchSample_Input input_batch)
        {
            int batchsize = input_batch.batchsize;
            for (int i = neurallinks.Count - 1; i > 0; i--)
            {
                NeuralLink nlink = neurallinks[i].NeuralLinkModel;
                if (nlink.Nt == N_Type.Fully_Connected)
                {
                    MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallayers[i + 1].ErrorDeriv, nlink.weight, neurallayers[i].ErrorDeriv, batchsize,
                            nlink.Neural_Out.Number, nlink.Neural_In.Number, 1);
                }
                else // must be composite full
                {
                    MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallayers[i + 1].ErrorDeriv, nlink.weight, neurallinks[i].CompError, batchsize,
                            nlink.Neural_Out.Number, (nlink.Neural_In.Number * nlink.N_Winsize + nlink.Extra_Input.Number), 1);
                    MathOperatorManager.GlobalInstance.FillOut_Composite(neurallayers[i].ErrorDeriv, input_batch, neurallinks[i].CompError, contextLT.Table, contextLT.InputDeriv,
                            nlink.Neural_In.Number, nlink.Extra_Input.Number, 0);
                }

                if (i != 1 || neurallinks[0].NeuralLinkModel.Nt != N_Type.bLSTM)
                {
                    if (neurallinks[i - 1].NeuralLinkModel.Af == A_Func.Tanh)
                    {
                        MathOperatorManager.GlobalInstance.Deriv_Tanh(neurallayers[i].ErrorDeriv, neurallayers[i].Output, batchsize, nlink.Neural_In.Number);
                    }
                    else if (neurallinks[i - 1].NeuralLinkModel.Af == A_Func.Rectified)
                    {
                        MathOperatorManager.GlobalInstance.Deriv_Rectified(neurallayers[i].ErrorDeriv, neurallayers[i].Output, batchsize, nlink.Neural_In.Number);
                    }
                }
            }

            if (neurallinks[0].NeuralLinkModel.Nt == N_Type.bLSTM)
            {
                // error back propagation in time
                MathOperatorManager.GlobalInstance.LSTM_Sequence_Backward(input_batch, neurallinks[0].NeuralLinkModel.recur_weight,
                    neurallinks[0].LayerMaxPooling_Index, neurallayers[1].ErrorDeriv, neurallinks[0].LayerPoolingA,
                    neurallinks[0].LayerPoolingI, neurallinks[0].LayerPoolingF, neurallinks[0].LayerPoolingO,
                    neurallinks[0].LayerPoolingInternalState, neurallinks[0].LayerPoolingOutput, neurallinks[0].NeuralLinkModel.Neural_Out.Number);
            }
        }

        unsafe public void backward_calculate_weight_deriv(BatchSample_Input input_batch) // 0-q1,1-q2,2-q3
        {
            //Calculate derivatives for all parameters, including link weight, bias and lookup tables. The derivatives for context table have already been computed 
            int batchsize = input_batch.batchsize;
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].ZeroDeriv();

                if (ParameterSetting.UpdateBias && neurallinks[i].NeuralLinkModel.Nt != N_Type.bLSTM)
                {
                    MathOperatorManager.GlobalInstance.Matrix_Aggragate_Sup(neurallayers[i + 1].ErrorDeriv, neurallinks[i].BiasDeriv, batchsize, neurallinks[i].NeuralLinkModel.Neural_Out.Number);
                }

                if (i == 0)
                {
                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.Fully_Connected)
                    {
                        throw new Exception("Not implemented! The first layer must be convolutional or multi-convolutional!");
                        //MathOperatorManager.GlobalInstance.SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(input_batch, neurallinks[i].WeightDeriv, neurallayers[i + 1].ErrorDeriv, neurallinks[i].NeuralLinkModel.Neural_In.Number, neurallinks[i].NeuralLinkModel.Neural_Out.Number, neurallinks[i].NeuralLinkModel.N_Winsize);
                    }
                    else if (neurallinks[i].NeuralLinkModel.Nt == N_Type.Convolution_layer)
                    {
                        MathOperatorManager.GlobalInstance.Convolution_Matrix_Product_Sup(neurallayers[i + 1].ErrorDeriv, neurallinks[i].LayerMaxPooling_Index, wordLT.Table,
                                     input_batch, neurallinks[i].NeuralLinkModel.N_Winsize,
                                     batchsize, neurallayers[i + 1].Number, neurallinks[i].WeightDeriv, neurallinks[i].NeuralLinkModel.Neural_In.Number);
                    }
                    else if (neurallinks[i].NeuralLinkModel.Nt == N_Type.bLSTM)
                    {
                        MathOperatorManager.GlobalInstance.LSTM_Weight_Deriv_Sup(input_batch, wordLT.Table, neurallinks[i].WeightDeriv,
                            neurallinks[i].LayerPoolingA, neurallinks[i].LayerPoolingI,
                            neurallinks[i].LayerPoolingF, neurallinks[i].LayerPoolingO,
                            neurallinks[i].LayerPoolingOutput, neurallinks[i].NeuralLinkModel.Neural_In.Number,
                            neurallinks[i].NeuralLinkModel.Neural_Out.Number, 0);
                        MathOperatorManager.GlobalInstance.LSTM_Weight_Deriv_Sup(input_batch, wordLT.Table, neurallinks[i].reWeightDeriv,
                            neurallinks[i].LayerPoolingA, neurallinks[i].LayerPoolingI,
                            neurallinks[i].LayerPoolingF, neurallinks[i].LayerPoolingO,
                            neurallinks[i].LayerPoolingOutput, neurallinks[i].NeuralLinkModel.Neural_In.Number,
                            neurallinks[i].NeuralLinkModel.Neural_Out.Number, 1);
                        MathOperatorManager.GlobalInstance.LSTM_Bias_Deriv_Sup(input_batch, neurallinks[i].BiasDeriv, neurallinks[i].LayerPoolingA, neurallinks[i].LayerPoolingI,
                            neurallinks[i].LayerPoolingF, neurallinks[i].LayerPoolingO, neurallinks[i].NeuralLinkModel.Neural_Out.Number);
                    }
                    else // must be multi-convolutional, composite layer cannot be the first layer
                    {
                        int accu = 0;
                        for (int b = 0; b < neurallinks[i].NeuralLinkModel.num_fms.Length; b++)
                        {
                            MathOperatorManager.GlobalInstance.MultiConv_Matrix_Product_Sup(neurallayers[i + 1].ErrorDeriv, neurallinks[i].LayerMaxPooling_Index, wordLT.Table,
                                     input_batch,
                                     batchsize, neurallayers[i + 1].Number, neurallinks[i].WeightDeriv, neurallinks[i].NeuralLinkModel.Neural_In.Number, neurallinks[i].NeuralLinkModel.wnd_sizes[b], neurallinks[i].NeuralLinkModel.num_fms[b], accu, b > 0 ? neurallinks[i].NeuralLinkModel.ma_sizes[b - 1] : 0);
                            accu += neurallinks[i].NeuralLinkModel.num_fms[b];
                        }
                    }
                }
                else
                {
                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.Fully_Connected)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Product_Sup(neurallayers[i].Output, neurallayers[i + 1].ErrorDeriv, neurallinks[i].WeightDeriv,
                                                batchsize, neurallayers[i].Number, neurallayers[i + 1].Number);
                    }
                    else // must be composite full
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Product_Sup(neurallinks[i].CompOutput, neurallayers[i + 1].ErrorDeriv, neurallinks[i].WeightDeriv,
                                                batchsize, (neurallayers[i].Number + neurallinks[i].NeuralLinkModel.Extra_Input.Number), neurallayers[i + 1].Number);
                    }

                }
            }
            //Finally, word vector derivatives
            wordLT.ZeroDeriv();
            if (neurallinks[0].NeuralLinkModel.Nt == N_Type.MultiWidthConv_layer)
            {
                    MathOperatorManager.GlobalInstance.MultiConv_Compute_WVDERIV(neurallayers[1].ErrorDeriv, neurallinks[0].LayerMaxPooling_Index, neurallinks[0].NeuralLinkModel.weight, batchsize,
                                neurallayers[1].Number, wordLT.InputDeriv, neurallinks[0].NeuralLinkModel.Neural_In.Number, neurallinks[0].NeuralLinkModel.winsizes, neurallinks[0].NeuralLinkModel.fmsizes);
            }
            else if (neurallinks[0].NeuralLinkModel.Nt == N_Type.bLSTM)
            {
                    MathOperatorManager.GlobalInstance.LSTM_Compute_WVDeriv(input_batch.elementSize, neurallinks[0].NeuralLinkModel.weight, wordLT.InputDeriv, neurallinks[0].LayerPoolingA,
                        neurallinks[0].LayerPoolingI, neurallinks[0].LayerPoolingF, neurallinks[0].LayerPoolingO, neurallinks[0].NeuralLinkModel.Neural_In.Number, neurallinks[0].NeuralLinkModel.Neural_Out.Number);
            }
            else // convolution
            {
                    MathOperatorManager.GlobalInstance.Conv_Compute_WVDERIV(neurallayers[1].ErrorDeriv, neurallinks[0].LayerMaxPooling_Index, neurallinks[0].NeuralLinkModel.weight, batchsize,
                                neurallayers[1].Number, wordLT.InputDeriv, neurallinks[0].NeuralLinkModel.Neural_In.Number, neurallinks[0].NeuralLinkModel.N_Winsize);
            }

            if (ParameterSetting.CheckGrad)
            {
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Update_Sup(wordLT.TabUpdate, wordLT, wordLT.Table.vecDim, 1.0f);
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Update_Sup(contextLT.TabUpdate, contextLT, contextLT.Table.vecDim, 1.0f);
            }
        }

        /// <summary>
        /// the error deriv at top output layer must have been set up before call this method.
        /// This process only do backprop computations. It does not update model weights at all.
        /// Need to call update_weight afterwards to update models.
        /// </summary>
        /// <param name="input_batch"></param>
        /// <param name="momentum"></param>
        /// <param name="learning_rate"></param>
        public void backward_propagate_deriv(BatchSample_Input input_batch)
        {
            // step 1, compute the derivatives for the output values of each layer
            backward_calculate_layerout_deriv(input_batch);
            // step 2, compute the derivatives for the connections of each neural link layer
            summarizeUnique(input_batch);
            backward_calculate_weight_deriv(input_batch);
        }

        /// <summary>
        /// Must call backward_propagate(), or two steps one by one, before calling this method.
        /// </summary>
        unsafe public void update_weight(BatchSample_Input input_batch, float momentum, float learning_rate)
        {
            /// First, update weights and bias
            /// step 1, compute the weight updates, taking momentum and learning rates into consideration
            /// Wei_Update = momentum * Wei_Update + learn_rate * grad.
            int row = 0, col = 0, rerow = 0, recol = 0;
            for (int i = 0; i < neurallinks.Count; i++)
            {
                int biassize = neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                if (neurallinks[i].NeuralLinkModel.Nt == N_Type.Composite_Full)
                {
                    row = neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                    col = neurallinks[i].NeuralLinkModel.Neural_In.Number * neurallinks[i].NeuralLinkModel.N_Winsize + neurallinks[i].NeuralLinkModel.Extra_Input.Number;
                }
                else if (neurallinks[i].NeuralLinkModel.Nt == N_Type.MultiWidthConv_layer)
                {
                    // treat feature dimensin as num of rows
                    row = neurallinks[i].NeuralLinkModel.Neural_In.Number;
                    col = neurallinks[i].NeuralLinkModel.ma_sizes[neurallinks[i].NeuralLinkModel.ma_sizes.Length - 1] / row;
                }
                else if (neurallinks[i].NeuralLinkModel.Nt == N_Type.bLSTM)
                {
                    row = neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                    col = neurallinks[i].NeuralLinkModel.Neural_In.Number * 4;
                    rerow = neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                    recol = neurallinks[i].NeuralLinkModel.Neural_Out.Number * 2;
                    biassize = neurallinks[i].NeuralLinkModel.Neural_Out.Number * 4;
                }
                else
                {
                    row = neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                    col = neurallinks[i].NeuralLinkModel.Neural_In.Number * neurallinks[i].NeuralLinkModel.N_Winsize;
                }

                if (ParameterSetting.updateScheme == 1)
                {
                    // add the momentum
                    MathOperatorManager.GlobalInstance.Scale_Matrix(neurallinks[i].WeightUpdate, col, row, momentum);

                    // dnn_neurallinks[i].Weight
                    MathOperatorManager.GlobalInstance.Matrix_Add(neurallinks[i].WeightUpdate, neurallinks[i].WeightDeriv, col, row, learning_rate);

                    // update the model: Weight = Weight += Wei_Update
                    MathOperatorManager.GlobalInstance.Matrix_Add_REAL(neurallinks[i].NeuralLinkModel.weight, neurallinks[i].WeightUpdate, col, row);

                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.bLSTM)
                    {
                        MathOperatorManager.GlobalInstance.Scale_Matrix(neurallinks[i].reWeightUpdate, recol, rerow, momentum);
                        MathOperatorManager.GlobalInstance.Matrix_Add(neurallinks[i].reWeightUpdate, neurallinks[i].reWeightDeriv, recol, rerow, learning_rate);
                        MathOperatorManager.GlobalInstance.Matrix_Add_REAL(neurallinks[i].NeuralLinkModel.recur_weight, neurallinks[i].reWeightUpdate, recol, rerow);
                    }
                }
                else if (ParameterSetting.updateScheme == 0)
                {
                    MathOperatorManager.GlobalInstance.Matrix_Grad_Decent(neurallinks[i].NeuralLinkModel.weight, neurallinks[i].WeightDeriv, col, row, learning_rate);
                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.bLSTM)
                        MathOperatorManager.GlobalInstance.Matrix_Grad_Decent(neurallinks[i].NeuralLinkModel.recur_weight, neurallinks[i].reWeightDeriv, recol, rerow, learning_rate);
                }
                else // AdaGrad
                {
                    MathOperatorManager.GlobalInstance.Matrix_Ada_Grad_Decent(neurallinks[i].NeuralLinkModel.weight, neurallinks[i].WeightDeriv, neurallinks[i].WeightAdaGrad, col, row, learning_rate, ParameterSetting.DSSMEpsilon);
                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.bLSTM)
                        MathOperatorManager.GlobalInstance.Matrix_Ada_Grad_Decent(neurallinks[i].NeuralLinkModel.recur_weight, neurallinks[i].reWeightDeriv, neurallinks[i].reWeightAdaGrad, recol, rerow, learning_rate, ParameterSetting.DSSMEpsilon);
                }

                if (ParameterSetting.UpdateBias)
                {
                    if (ParameterSetting.updateScheme == 1)
                    {
                        // add the momentum
                        MathOperatorManager.GlobalInstance.Scale_Matrix(neurallinks[i].BiasUpdate, 1, biassize, momentum);

                        // dnn_neurallinks[i].Weight
                        MathOperatorManager.GlobalInstance.Matrix_Add(neurallinks[i].BiasUpdate, neurallinks[i].BiasDeriv, 1, biassize, learning_rate);
                        // upate the model
                        MathOperatorManager.GlobalInstance.Matrix_Add_REAL(neurallinks[i].NeuralLinkModel.bias, neurallinks[i].BiasUpdate, 1, biassize);
                    }
                    else if (ParameterSetting.updateScheme == 0)
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Grad_Decent(neurallinks[i].NeuralLinkModel.bias, neurallinks[i].BiasDeriv, 1, biassize, learning_rate);
                    }
                    else // AdaGrad
                    {
                        MathOperatorManager.GlobalInstance.Matrix_Ada_Grad_Decent(neurallinks[i].NeuralLinkModel.bias, neurallinks[i].BiasDeriv, neurallinks[i].BiasAdaGrad, 1, biassize, learning_rate, ParameterSetting.DSSMEpsilon);
                    }
                }
            }

            // update lookup tables
            if (ParameterSetting.updateScheme == 0)
            {
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Sup(wordLT.Table, wordLT, wordLT.Table.vecDim, learning_rate);
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Sup(contextLT.Table, contextLT, contextLT.Table.vecDim, learning_rate);
            }
            else if (ParameterSetting.updateScheme == 1)
            {
                MathOperatorManager.GlobalInstance.Scale_Matrix(wordLT.TabUpdate, wordLT.Table.count, wordLT.Table.vecDim, momentum);
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Update_Sup(wordLT.TabUpdate, wordLT, wordLT.Table.vecDim, learning_rate);
                MathOperatorManager.GlobalInstance.Matrix_Add_REAL(wordLT.Table.table, wordLT.TabUpdate, wordLT.Table.count, wordLT.Table.vecDim);

                MathOperatorManager.GlobalInstance.Scale_Matrix(contextLT.TabUpdate, contextLT.Table.count, contextLT.Table.vecDim, momentum);
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Update_Sup(contextLT.TabUpdate, contextLT, contextLT.Table.vecDim, learning_rate);
                MathOperatorManager.GlobalInstance.Matrix_Add_REAL(contextLT.Table.table, contextLT.TabUpdate, contextLT.Table.count, contextLT.Table.vecDim);
            }
            else // AdaGrad
            {
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Ada_Sup(wordLT.Table, wordLT, wordLT.Table.vecDim, learning_rate, ParameterSetting.DSSMEpsilon);
                MathOperatorManager.GlobalInstance.Sparse_Update_Lookup_Ada_Sup(contextLT.Table, contextLT, contextLT.Table.vecDim, learning_rate, ParameterSetting.DSSMEpsilon);
            }

        }

        public void Dispose()
        {
            if (neurallayers != null)
            {
                for (int i = 0; i < neurallayers.Count; i++)
                    neurallayers[i].Dispose();
                neurallayers.Clear();
            }
            if (neurallinks != null)
            {
                for (int i = 0; i < neurallinks.Count; i++)
                    neurallinks[i].Dispose();
                neurallinks.Clear();
            }
            if (wordLT != null)
            {
                wordLT.Dispose();
                wordLT = null;
            }
            if (contextLT != null)
            {
                contextLT.Dispose();
                contextLT = null;
            }
        }

        unsafe void summarizeUnique(BatchSample_Input inputBatch)
        {
            Dictionary<int, List<int>> wordSum = new Dictionary<int, List<int>>(10000);
            
            
            for (int j = 0; j < inputBatch.elementSize; j++)
            {
                int w = inputBatch.Word_Idx_Mem[j];
                if (wordSum.ContainsKey(w))
                    wordSum[w].Add(j);
                else
                {
                    wordSum.Add(w, new List<int>());
                    wordSum[w].Add(j);
                }
            }
            
            int c1 = 0, c2 = 0;
            foreach (KeyValuePair<int, List<int>> d in wordSum)
            {
                wordLT.uniqueWordID.MemPtr[c1] = d.Key;
                for (int i = 0; i < d.Value.Count; i++)
                {
                    wordLT.Sequence.MemPtr[c2] = d.Value[i];
                    c2++;
                }
                wordLT.uniqueWordIdx.MemPtr[c1] = c2;
                c1++;
            }
            wordLT.uniqueNum = c1;

            wordSum.Clear();
            
            
            for (int j = 0; j < inputBatch.batchsize; j++)
            {
                int w = inputBatch.Fea_Mem[j];
                if (wordSum.ContainsKey(w))
                    wordSum[w].Add(j);
                else
                {
                    wordSum.Add(w, new List<int>());
                    wordSum[w].Add(j);
                }
            }
            
            c1 = 0;
            c2 = 0;
            foreach (KeyValuePair<int, List<int>> d in wordSum)
            {
                contextLT.uniqueWordID.MemPtr[c1] = d.Key;
                for (int i = 0; i < d.Value.Count; i++)
                {
                    contextLT.Sequence.MemPtr[c2] = d.Value[i];
                    c2++;
                }
                contextLT.uniqueWordIdx.MemPtr[c1] = c2;
                c1++;
            }
            contextLT.uniqueNum = c1;

            wordLT.uniqueWordID.CopyIntoCuda();
            wordLT.uniqueWordIdx.CopyIntoCuda();
            wordLT.Sequence.CopyIntoCuda();

            contextLT.uniqueWordID.CopyIntoCuda();
            contextLT.uniqueWordIdx.CopyIntoCuda();
            contextLT.Sequence.CopyIntoCuda();
        }
    }

    public class LookupTabRunDataSup : IDisposable
    {
        LookupTab table;
        public LookupTab Table { get { return table; } }

        public int Dim { get { return table.vecDim; } }
        public int Count { get { return table.count; } }
        bool isWordInput; // if wordInput, use MAXSEGMENT_BATCH to construct InputDeriv

        CudaPieceFloat inputDeriv = null;
        CudaPieceFloat tabUpdate = null;
        CudaPieceFloat tabAdaGrad = null;

        public CudaPieceFloat InputDeriv { get { return inputDeriv; } }
        public CudaPieceFloat TabUpdate { get { return tabUpdate; } }
        public CudaPieceFloat TabAdaGrad { get { return tabAdaGrad; } }

        //some auxiliary variables
        public CudaPieceInt uniqueWordID = null;
        public CudaPieceInt uniqueWordIdx = null;
        public CudaPieceInt Sequence = null;
        public int uniqueNum = 0;

        // improve, do not read gloabl MAXSEGMENT_BATCH, take it as an argument
        public LookupTabRunDataSup(LookupTab table, bool isWordInput, int maxSeg_batch, int maxBatchsize)
        {
            this.table = table;
            this.isWordInput = isWordInput;

            
            if (ParameterSetting.updateScheme == 1 || ParameterSetting.CheckGrad)
                tabUpdate = new CudaPieceFloat(Dim * Count, ParameterSetting.CheckGrad ? true : false, true);
            else if (ParameterSetting.updateScheme == 2)
                tabAdaGrad = new CudaPieceFloat(Dim * Count, false, true);
            if (isWordInput)
            {
                inputDeriv = new CudaPieceFloat(Dim * maxSeg_batch, false, true);
                Sequence = new CudaPieceInt(maxSeg_batch, true, true);
            }
            else
            {
                inputDeriv = new CudaPieceFloat(Dim * maxBatchsize, false, true);
                Sequence = new CudaPieceInt(maxBatchsize, true, true);
            }
            uniqueWordID = new CudaPieceInt(Count, true, true);
            uniqueWordIdx = new CudaPieceInt(Count, true, true);

        }

        ~LookupTabRunDataSup()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (inputDeriv != null)
            {
                inputDeriv.Dispose();
                inputDeriv = null;
            }
            if (tabUpdate != null)
            {
                tabUpdate.Dispose();
                tabUpdate = null;
            }
            if (uniqueWordID != null)
                uniqueWordID.Dispose();
            if (uniqueWordIdx != null)
                uniqueWordIdx.Dispose();
            if (Sequence != null)
                Sequence.Dispose();
            if (tabAdaGrad != null)
                tabAdaGrad.Dispose();
        }

        public void ZeroDeriv()
        {
            if (inputDeriv != null)
            {
                inputDeriv.Zero();
            }
        }

    }

    /// <summary>
    /// A particular run related data for supervised training and normal forward propagation, including feedforward outputs and backward propagation derivatives
    /// </summary>
    public class NeuralLayerDataSup : IDisposable
    {
        NeuralLayer LayerModel;
        public int Number { get { return LayerModel.Number; } }
        /// <summary>        
        /// </summary>
        /// <param name="num"></param>
        /// <param name="isValueNeeded">To save GPU memory, when no errors are needed, we should not allocate error piece. This usually happens on the input layer</param>
        public NeuralLayerDataSup(NeuralLayer layerModel, int maxBatchsize, bool isValueNeeded, bool isForwardOnly, bool isLast)
        {
            LayerModel = layerModel;
            if (isValueNeeded)
            {
                output = new CudaPieceFloat(maxBatchsize * Number, isLast || ParameterSetting.DEBUG, true);
                if (!isForwardOnly)
                    errorDeriv = new CudaPieceFloat(maxBatchsize * Number, isLast || ParameterSetting.DEBUG, true);
            }
        }
        /// <summary>
        /// The output of the layer, i.e., the actual activitation values
        /// </summary>
        CudaPieceFloat output = null;

        public CudaPieceFloat Output
        {
            get { return output; }
        }
        /// <summary>
        /// The error of the layer, back-propagated from the top loss function
        /// </summary>
        CudaPieceFloat errorDeriv = null;

        public CudaPieceFloat ErrorDeriv
        {
            get { return errorDeriv; }
        }


        ~NeuralLayerDataSup()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (output != null)
            {
                output.Dispose();
                output = null;
            }
            if (errorDeriv != null)
            {
                errorDeriv.Dispose();
                errorDeriv = null;
            }
        }
    }
    /// <summary>
    /// A particular run related data for supervised training and normal forward propagation, including feedforward outputs and backward propagation derivatives
    /// </summary>
    public class NeuralLinkDataSup : IDisposable
    {
        NeuralLink neuralLinkModel;

        public NeuralLink NeuralLinkModel
        {
            get { return neuralLinkModel; }
        }

        /// <summary>
        /// Used if convolutional
        /// </summary>
        CudaPieceFloat layerPoolingOutput = null;

        public CudaPieceFloat LayerPoolingOutput
        {
            get { return layerPoolingOutput; }
        }
        /// <summary>
        /// Used if convolutional and maxpooling
        /// </summary>
        CudaPieceInt layerMaxPooling_Index = null;

        public CudaPieceInt LayerMaxPooling_Index
        {
            get { return layerMaxPooling_Index; }
        }

        // for LSTM exclusively, storing internal state c (forward) or derivative of c (backward)
        CudaPieceFloat layerPoolingInternalState = null;
        public CudaPieceFloat LayerPoolingInternalState
        {
            get { return layerPoolingInternalState; }
        }

        // for LSTM exclusively, storing activations(forward)/derivatives(backward) of internal input a, a =  tanh(W_a *x + U_a * h + b_a)
        CudaPieceFloat layerPoolingA = null;
        public CudaPieceFloat LayerPoolingA
        {
            get { return layerPoolingA; }
        }

        // for LSTM exclusively, storing activations(forward)/derivatives(backward) of internal input gates
        CudaPieceFloat layerPoolingI = null;
        public CudaPieceFloat LayerPoolingI
        {
            get { return layerPoolingI; }
        }

        // for LSTM exclusively, storing activations(forward)/derivatives(backward) of internal forget gates
        CudaPieceFloat layerPoolingF = null;
        public CudaPieceFloat LayerPoolingF
        {
            get { return layerPoolingF; }
        }

        // for LSTM exclusively, storing activations(forward)/derivatives(backward) of internal output gates
        CudaPieceFloat layerPoolingO = null;
        public CudaPieceFloat LayerPoolingO
        {
            get { return layerPoolingO; }
        }

        CudaPieceFloat weightDeriv = null;

        public CudaPieceFloat WeightDeriv
        {
            get { return weightDeriv; }
        }

        CudaPieceFloat reweightDeriv = null;
        public CudaPieceFloat reWeightDeriv
        {
            get { return reweightDeriv; }
        }

        CudaPieceFloat biasDeriv = null;

        public CudaPieceFloat BiasDeriv
        {
            get { return biasDeriv; }
        }

        /// <summary>
        /// Output cache for composite layer
        /// </summary>
        CudaPieceFloat composite_output = null;

        public CudaPieceFloat CompOutput
        {
            get { return composite_output; }
        }

        /// <summary>
        /// composite error derivatives
        /// </summary>
        CudaPieceFloat composite_error = null;

        public CudaPieceFloat CompError
        {
            get { return composite_error; }
        }

        /// <summary>
        /// Wei_Update = momentum * Wei_Update + learn_rate * grad.
        /// Weight = Weight + Wei_Update
        /// </summary>
        CudaPieceFloat weightUpdate = null;

        public CudaPieceFloat WeightUpdate
        {
            get { return weightUpdate; }
        }

        CudaPieceFloat reweightUpdate = null;
        public CudaPieceFloat reWeightUpdate
        {
            get { return reweightUpdate; }
        }

        CudaPieceFloat biasUpdate = null;

        public CudaPieceFloat BiasUpdate
        {
            get { return biasUpdate; }
        }

        // used by adaptive gradient descent  adagrad += grad^2;    weight = weight - lr*grad/sqrt(adagrad)
        CudaPieceFloat weightAdaGrad = null;

        public CudaPieceFloat WeightAdaGrad
        {
            get { return weightAdaGrad; }
        }

        CudaPieceFloat reweightAdaGrad = null;

        public CudaPieceFloat reWeightAdaGrad
        {
            get { return reweightAdaGrad; }
        }

        CudaPieceFloat biasAdaGrad = null;

        public CudaPieceFloat BiasAdaGrad
        {
            get { return biasAdaGrad; }
        }

        public NeuralLinkDataSup(NeuralLink neuralLink, int maxBatchsize, int maxSegsize, bool isForwardOnly)
        {
            neuralLinkModel = neuralLink;

            if (neuralLinkModel.Nt == N_Type.Convolution_layer || neuralLinkModel.Nt == N_Type.MultiWidthConv_layer || neuralLinkModel.Nt == N_Type.bLSTM)
            {
                // **now has the same shape as in single convolution, but some cells are invalid** for multi-window case, we still allocate the same size of space, but pooling output stores three matrices, each of size (seg_size - batchsize*(win_size-1))* fm_size
                layerPoolingOutput = new CudaPieceFloat(maxSegsize * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG, true);
                layerMaxPooling_Index = new CudaPieceInt(maxBatchsize * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG, true);

                if (neuralLinkModel.Nt == N_Type.bLSTM)
                {
                    layerPoolingInternalState = new CudaPieceFloat(maxSegsize * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG, true);
                    layerPoolingA = new CudaPieceFloat(maxSegsize * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG, true);
                    layerPoolingI = new CudaPieceFloat(maxSegsize * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG, true);
                    layerPoolingF = new CudaPieceFloat(maxSegsize * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG, true);
                    layerPoolingO = new CudaPieceFloat(maxSegsize * neuralLinkModel.Neural_Out.Number, ParameterSetting.DEBUG, true);
                }
            }

            int totalweightsize = 0;
            int totalreweightsize = 0;
            int totalbiassize = neuralLinkModel.Neural_Out.Number;
            if (neuralLinkModel.Nt == N_Type.MultiWidthConv_layer)
            {
                for (int i = 0; i < neuralLinkModel.wnd_sizes.Length; i++)
                    totalweightsize += neuralLinkModel.Neural_In.Number * neuralLinkModel.wnd_sizes[i] * neuralLinkModel.num_fms[i];
            }
            else if (neuralLinkModel.Nt == N_Type.Composite_Full)
            {
                totalweightsize = neuralLinkModel.Neural_Out.Number * (neuralLinkModel.Neural_In.Number + neuralLinkModel.Extra_Input.Number);
                // Create space for composite outputs and their derivatives
                composite_output = new CudaPieceFloat((neuralLinkModel.Neural_In.Number * neuralLinkModel.N_Winsize + neuralLinkModel.Extra_Input.Number) * maxBatchsize, ParameterSetting.DEBUG, true);
                if (!isForwardOnly)
                    composite_error = new CudaPieceFloat((neuralLinkModel.Neural_In.Number * neuralLinkModel.N_Winsize + neuralLinkModel.Extra_Input.Number) * maxBatchsize, ParameterSetting.DEBUG, true);
              
            }
            else if (neuralLinkModel.Nt == N_Type.bLSTM)
            {
                // calculate the size of weight and recurrent weight
                totalweightsize = neuralLinkModel.Neural_In.Number * (neuralLinkModel.Neural_Out.Number / 2) * 4 * 2;
                totalreweightsize = (neuralLinkModel.Neural_Out.Number / 2) * (neuralLinkModel.Neural_Out.Number / 2) * 4 * 2;
                totalbiassize = (neuralLinkModel.Neural_Out.Number / 2) * 4 * 2;
            }
            else
                totalweightsize = neuralLinkModel.Neural_In.Number * neuralLinkModel.Neural_Out.Number * neuralLinkModel.N_Winsize;

            if (!isForwardOnly)
            {
                weightDeriv = new CudaPieceFloat(totalweightsize, ParameterSetting.CheckGrad ? true : false, true);
                biasDeriv = new CudaPieceFloat(totalbiassize, ParameterSetting.CheckGrad ? true : false, true);
                if (neuralLinkModel.Nt == N_Type.bLSTM)
                    reweightDeriv = new CudaPieceFloat(totalreweightsize, ParameterSetting.CheckGrad ? true : false, true);

                if (ParameterSetting.updateScheme == 1)
                {
                    weightUpdate = new CudaPieceFloat(totalweightsize, false, true);
                    biasUpdate = new CudaPieceFloat(totalbiassize, false, true);
                    if (neuralLinkModel.Nt == N_Type.bLSTM)
                        reweightUpdate = new CudaPieceFloat(totalreweightsize, false, true);
                }
                else if (ParameterSetting.updateScheme == 2)
                {
                    weightAdaGrad = new CudaPieceFloat(totalweightsize, false, true, ParameterSetting.initAdaGrad);
                    biasAdaGrad = new CudaPieceFloat(totalbiassize, false, true, ParameterSetting.initAdaGrad);
                    if (neuralLinkModel.Nt == N_Type.bLSTM)
                        reweightAdaGrad = new CudaPieceFloat(totalreweightsize, false, true, ParameterSetting.initAdaGrad);
                }
            }
        }

        ~NeuralLinkDataSup()
        {
            Dispose();
        }


        public void Dispose()
        {
            if (layerPoolingOutput != null)
            {
                layerPoolingOutput.Dispose();
                layerPoolingOutput = null;
            }
            if (layerMaxPooling_Index != null)
            {
                layerMaxPooling_Index.Dispose();
                layerMaxPooling_Index = null;
            }

            if (composite_output != null)
            {
                composite_output.Dispose();
                composite_output = null;
            }
            if (composite_error != null)
            {
                composite_error.Dispose();
                composite_error = null;
            }

            if (weightDeriv != null)
            {
                weightDeriv.Dispose();
            }
            if (reweightDeriv != null)
            {
                reweightDeriv.Dispose();
            }
            if (biasDeriv != null)
            {
                biasDeriv.Dispose();
            }
            
            if (weightUpdate != null)
            {
                weightUpdate.Dispose();
            }
            if (reweightUpdate != null)
            {
                reweightUpdate.Dispose();
            }
            if (biasUpdate != null)
            {
                biasUpdate.Dispose();
            }

            if (weightAdaGrad != null)
                weightAdaGrad.Dispose();
            if (reweightAdaGrad != null)
                reweightAdaGrad.Dispose();
            if (biasAdaGrad != null)
                biasAdaGrad.Dispose();

            if (layerPoolingInternalState != null)
            {
                layerPoolingInternalState.Dispose();
                layerPoolingInternalState = null;
            }
            if (layerPoolingA != null)
            {
                layerPoolingA.Dispose();
                layerPoolingA = null;
            }
            if (layerPoolingI != null)
            {
                layerPoolingI.Dispose();
                layerPoolingI = null;
            }
            if (layerPoolingF != null)
            {
                layerPoolingF.Dispose();
                layerPoolingF = null;
            }
            if (layerPoolingO != null)
            {
                layerPoolingO.Dispose();
                layerPoolingO = null;
            }
        }

        public void ZeroDeriv()
        {
            if (weightDeriv != null)
                weightDeriv.Zero();
            if (biasDeriv != null)
                biasDeriv.Zero();
            if (neuralLinkModel.Nt == N_Type.bLSTM)
                reweightDeriv.Zero();
        }
    }
}
