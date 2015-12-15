using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DSMlib
{
   
    public class BatchSample_Input : IDisposable
    {
        public int batchsize = ParameterSetting.BATCH_SIZE;
        CudaPieceInt word_Idx;
        CudaPieceInt sample_Idx;

        public int[] Word_Idx_Mem { get { return word_Idx.MemPtr; } }
        public int[] Sample_Mem { get { return sample_Idx.MemPtr; } }

        public IntPtr Word_Idx { get { return word_Idx.CudaPtr; } }
        public IntPtr Sample_Idx { get { return sample_Idx.CudaPtr; } }


        public BatchSample_Input(int MAX_BATCH_SIZE, int MAXELEMENTS_PERBATCH)
        {
            sample_Idx = new CudaPieceInt(MAX_BATCH_SIZE, true, true);
            word_Idx = new CudaPieceInt(MAXELEMENTS_PERBATCH, true, true);
            
        }
        ~BatchSample_Input()
        {
            Dispose();
        }


        public void Load(BinaryReader mreader, int batchsize)
        {
            int elementSize = mreader.ReadInt32();
            for (int i = 0; i < batchsize; i++)  //read sample index.
            {
                Sample_Mem[i] = mreader.ReadInt32();
            }

            // update cudaDataPiece sizes
            sample_Idx.Size = batchsize;
            word_Idx.Size = elementSize;

            for (int i = 0; i < elementSize; i++)
            {
                Word_Idx_Mem[i] = mreader.ReadInt32();
            }
        }

        public void Batch_In_GPU()
        {
            sample_Idx.CopyIntoCuda();
            word_Idx.CopyIntoCuda();
        }

        public void Dispose()
        {
            sample_Idx.Dispose();
            word_Idx.Dispose();

        }
    }

}
