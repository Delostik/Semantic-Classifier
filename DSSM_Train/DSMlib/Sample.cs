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
        public int elementSize;
        protected CudaPieceInt word_Idx;
        protected CudaPieceInt sample_Idx;
        protected CudaPieceInt fea_Idx;
        protected CudaPieceInt seg_Margin;

        public int[] Fea_Mem { get { return fea_Idx.MemPtr; } }
        public int[] Sample_Mem { get { return sample_Idx.MemPtr; } }
        public int[] Word_Idx_Mem { get { return word_Idx.MemPtr; } }
        public int[] Seg_Margin_Mem { get { return seg_Margin.MemPtr;  } }

        public IntPtr Fea_Idx { get { return fea_Idx.CudaPtr; } }
        public IntPtr Sample_Idx { get { return sample_Idx.CudaPtr; } }
        public IntPtr Word_Idx { get { return word_Idx.CudaPtr; } }
        public IntPtr Seg_Margin { get { return seg_Margin.CudaPtr; } }

        public BatchSample_Input(int MAX_BATCH_SIZE, int MAXELEMENTS_PERBATCH)
        {
            sample_Idx = new CudaPieceInt(MAX_BATCH_SIZE, true, true);
            fea_Idx = new CudaPieceInt(MAX_BATCH_SIZE, true, true);
            word_Idx = new CudaPieceInt(MAXELEMENTS_PERBATCH, true, true);
            seg_Margin = new CudaPieceInt(MAXELEMENTS_PERBATCH, true, true);
        }

        ~BatchSample_Input()
        {
            Dispose();
        }

        public virtual void Load(BinaryReader mreader, int batchsize)
        {
            elementSize = mreader.ReadInt32();
            this.batchsize = mreader.ReadInt32();
            if (this.batchsize != batchsize)
            {
                throw new Exception(string.Format(
                    "Batch_Size does not match between configuration and input data!\n\tFrom config: {0}.\n\tFrom data: {1}"
                    , batchsize, this.batchsize)
                );
            }
            for (int i = 0; i < batchsize; i++)  //read sample index.
            {
                Sample_Mem[i] = mreader.ReadInt32();
            }
            for (int i = 0; i < batchsize; i++)
            {
                Fea_Mem[i] = mreader.ReadInt32();
            }
            // update cudaDataPiece sizes
            sample_Idx.Size = batchsize;
            fea_Idx.Size = batchsize;
            word_Idx.Size = elementSize;
            seg_Margin.Size = elementSize;

            int smp_index = 0;
            for (int i = 0; i < elementSize; i++)
            {
                Word_Idx_Mem[i] = mreader.ReadInt32();
                while (Sample_Mem[smp_index] <= i)
                {
                    smp_index++;
                }
                Seg_Margin_Mem[i] = smp_index;
            }
        }

        public virtual void Batch_In_GPU()
        {
            sample_Idx.CopyIntoCuda();
            fea_Idx.CopyIntoCuda();
            word_Idx.CopyIntoCuda();
            seg_Margin.CopyIntoCuda();
        }

        public void Dispose()
        {
            if (sample_Idx != null)
            {
                sample_Idx.Dispose();
                sample_Idx = null;
            }
            if (fea_Idx != null)
            {
                fea_Idx.Dispose();
                fea_Idx = null;
            }
            if (word_Idx != null)
            {
                word_Idx.Dispose();
                word_Idx = null;
            }
            if (seg_Margin != null)
            {
                seg_Margin.Dispose();
                seg_Margin = null;
            }
        }
    }

    public class LabeledBatchSample_Input : BatchSample_Input
    {
        protected CudaPieceInt emo_Idx;
        protected CudaPieceInt subj_Idx;
        
        public int[] Emo_Mem { get { return emo_Idx.MemPtr; } }
        
        public IntPtr Emo_Idx { get { return emo_Idx.CudaPtr; } }

        public int[] Subj_Mem { get { return subj_Idx.MemPtr; } }

        public IntPtr Subj_Idx { get { return subj_Idx.CudaPtr; } }
        

        public LabeledBatchSample_Input(int MAX_BATCH_SIZE, int MAXELEMENTS_PERBATCH) : base(MAX_BATCH_SIZE, MAXELEMENTS_PERBATCH)
        {
            emo_Idx = new CudaPieceInt(MAX_BATCH_SIZE, true, true);
            subj_Idx = new CudaPieceInt(MAX_BATCH_SIZE, true, false);
        }

        ~LabeledBatchSample_Input()
        {
            Dispose();
        }

        public override void Load(BinaryReader mreader, int batchsize)
        {
            this.elementSize = mreader.ReadInt32();
            this.batchsize = mreader.ReadInt32();
            if (batchsize != this.batchsize)
            {
                throw new Exception(string.Format(
                    "Batch_Size does not match between configuration and input data!\n\tFrom config: {0}.\n\tFrom data: {1}"
                    , batchsize, this.batchsize)
                );
            }
            for (int i = 0; i < batchsize; i++)  //read sample index.
            {
                Sample_Mem[i] = mreader.ReadInt32();
            }
            for (int i = 0; i < batchsize; i++)
            {
                Fea_Mem[i] = mreader.ReadInt32();
            }
            for (int i = 0; i < batchsize; i++)
            {
                Emo_Mem[i] = mreader.ReadInt32();
            }
            for (int i = 0; i < batchsize; i++)
            {
                Subj_Mem[i] = mreader.ReadInt32();
            }
            // update cudaDataPiece sizes
            sample_Idx.Size = batchsize;
            fea_Idx.Size = batchsize;
            emo_Idx.Size = batchsize;
            subj_Idx.Size = batchsize;
            word_Idx.Size = elementSize;
            seg_Margin.Size = elementSize;

            int smp_index = 0;
            for (int i = 0; i < elementSize; i++)
            {
                Word_Idx_Mem[i] = mreader.ReadInt32();
                while (Sample_Mem[smp_index] <= i)
                {
                    smp_index++;
                }
                Seg_Margin_Mem[i] = smp_index;
            }
        }

        public override void Batch_In_GPU()
        {
            sample_Idx.CopyIntoCuda();
            fea_Idx.CopyIntoCuda();
            emo_Idx.CopyIntoCuda();
            word_Idx.CopyIntoCuda();
            seg_Margin.CopyIntoCuda();
        }

        public void Dispose()
        {
            if (sample_Idx != null)
            {
                sample_Idx.Dispose();
                sample_Idx = null;
            }
            if (fea_Idx != null)
            {
                fea_Idx.Dispose();
                fea_Idx = null;
            }
            if (word_Idx != null)
            {
                word_Idx.Dispose();
                word_Idx = null;
            }
            if (seg_Margin != null)
            {
                seg_Margin.Dispose();
                seg_Margin = null;
            }
            if (emo_Idx != null)
            {
                emo_Idx.Dispose();
                emo_Idx = null;
            }
            if (subj_Idx != null)
            {
                subj_Idx.Dispose();
                subj_Idx = null;
            }
        }
    }
}
