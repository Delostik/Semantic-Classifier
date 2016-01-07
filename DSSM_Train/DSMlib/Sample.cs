﻿using System;
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
        CudaPieceInt word_Idx;
        CudaPieceInt sample_Idx;
        CudaPieceInt fea_Idx;
        CudaPieceInt seg_Margin;

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

        public void Load(BinaryReader mreader, int batchsize)
        {
            this.batchsize = batchsize;
            elementSize = mreader.ReadInt32();
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

        public void Batch_In_GPU()
        {
            sample_Idx.CopyIntoCuda();
            fea_Idx.CopyIntoCuda();
            word_Idx.CopyIntoCuda();
            seg_Margin.CopyIntoCuda();
        }

        public void Dispose()
        {
            sample_Idx.Dispose();
            fea_Idx.Dispose();
            word_Idx.Dispose();
            seg_Margin.Dispose();
        }
    }

    public class LabeledBatchSample_Input : IDisposable
    {
        public int batchsize = ParameterSetting.BATCH_SIZE;
        CudaPieceInt word_Idx;
        CudaPieceInt sample_Idx;
        CudaPieceInt fea_Idx;
        CudaPieceInt emo_Idx;
        CudaPieceInt seg_Margin;

        public int[] Fea_Mem { get { return fea_Idx.MemPtr; } }
        public int[] Emo_Mem { get { return emo_Idx.MemPtr; } }
        public int[] Sample_Mem { get { return sample_Idx.MemPtr; } }
        public int[] Word_Idx_Mem { get { return word_Idx.MemPtr; } }
        public int[] Seg_Margin_Mem { get { return seg_Margin.MemPtr; } }

        public IntPtr Fea_Idx { get { return fea_Idx.CudaPtr; } }
        public IntPtr Emo_Idx { get { return emo_Idx.CudaPtr; } }
        public IntPtr Sample_Idx { get { return sample_Idx.CudaPtr; } }
        public IntPtr Word_Idx { get { return word_Idx.CudaPtr; } }
        public IntPtr Seg_Margin { get { return seg_Margin.CudaPtr; } }

        public LabeledBatchSample_Input(int MAX_BATCH_SIZE, int MAXELEMENTS_PERBATCH)
        {
            sample_Idx = new CudaPieceInt(MAX_BATCH_SIZE, true, true);
            fea_Idx = new CudaPieceInt(MAX_BATCH_SIZE, true, true);
            emo_Idx = new CudaPieceInt(MAX_BATCH_SIZE, true, true);
            word_Idx = new CudaPieceInt(MAXELEMENTS_PERBATCH, true, true);
            seg_Margin = new CudaPieceInt(MAXELEMENTS_PERBATCH, true, true);
        }

        ~LabeledBatchSample_Input()
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
            for (int i = 0; i < batchsize; i++)
            {
                Fea_Mem[i] = mreader.ReadInt32();
            }
            for (int i = 0; i < batchsize; i++)
            {
                Emo_Mem[i] = mreader.ReadInt32();
            }
            // update cudaDataPiece sizes
            sample_Idx.Size = batchsize;
            fea_Idx.Size = batchsize;
            emo_Idx.Size = batchsize;
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

        public void Batch_In_GPU()
        {
            sample_Idx.CopyIntoCuda();
            fea_Idx.CopyIntoCuda();
            emo_Idx.CopyIntoCuda();
            word_Idx.CopyIntoCuda();
            seg_Margin.CopyIntoCuda();
        }

        public void Dispose()
        {
            sample_Idx.Dispose();
            fea_Idx.Dispose();
            emo_Idx.Dispose();
            word_Idx.Dispose();
            seg_Margin.Dispose();
        }
    }
}
