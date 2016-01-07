using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DSMlib
{
    interface IMathOperationManager
    {
        /*----------  Mainly used in forward activation, computing linear transformations  --------------*/
        //void SEQ_Sparse_Matrix_Multiply_INTEX(BatchSample_Input data, CudaPieceFloat weight, CudaPieceFloat output, int inputDimension, int outputDimension, int winSize);
        void Convolution_Matrix_Multiply_INTEX(BatchSample_Input data, CudaPieceFloat weight, CudaPieceFloat layerPoolingOutput, LookupTab wtab, int inputDimension, int outputDimension, int winSize) ;
        void MultiConv_Matrix_Multiply_INTEX(BatchSample_Input data, CudaPieceFloat weight, CudaPieceFloat layerPoolingOutput, LookupTab wtab, int inputDimension, int outputDimension, CudaPieceInt wndSizes, CudaPieceInt fmSizes);
        void Max_Pooling(CudaPieceFloat layerPoolingOutput, BatchSample_Input data, CudaPieceFloat output, CudaPieceInt layerMaxPooling_Index, int outputDimension, int winSize);
        void Multi_Max_Pooling(CudaPieceFloat layerPoolingOutput, BatchSample_Input data, CudaPieceFloat output, CudaPieceInt layerMaxPooling_Index, int outputDimension, CudaPieceInt wndSizes, CudaPieceInt fmSizes);
        void Matrix_Multipy(CudaPieceFloat input, CudaPieceFloat weight, CudaPieceFloat output, int batchsize, int inputDimension, int outputDimension, int inverse);
        
        /*----------  Mainly used in forward activation, adding non-linear activations  --------------*/
        void Matrix_Add_Tanh(CudaPieceFloat output, CudaPieceFloat bias, int batchsize, int outputDimension);
        void Matrix_Add_Vector(CudaPieceFloat output, CudaPieceFloat bias, int batchsize, int outputDimension);
        void Matrix_Rectified_Vector(CudaPieceFloat output, CudaPieceFloat bias, int batchsize, int outputDimension);

        /*----------  Mainly used in forward activation, computing loss function   ---------------*/
        void Cosine_Similarity(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int mIndex, int batchsize, int topLayerSize, float eps); //float.Epsilon);
        void Cosine_Similarity_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int mIndex, int batchsize, int outputLayerSize, float eps); //float.Epsilon);
        void Calculate_Alpha(CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA);
        void Calculate_Alpha_MXE(CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA);
        void Calculate_Alpha_NCE(CudaPieceFloat alpha, CudaPieceFloat dist, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA);
        void Calculate_Alpha_NCE2(CudaPieceFloat alpha, CudaPieceFloat dist, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA);
        void Calculate_Alpha_PAIRRANK(CudaPieceFloat alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma);
        void FillOut_Dist_NCE(CudaPieceFloat dist, CudaPieceInt GPU_negative_index, int nTrailPlus1, int BATCH_SIZE, int mIndex, int batchsize);
        
        /*----------  Mainly used in backward propagation, computing top layer error derivative --------------*/
        void Deriv_Cosine(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps);
        void Deriv_Dis(CudaPieceFloat s1deriv, CudaPieceFloat s2deriv, CudaPieceFloat s3deriv, CudaPieceFloat s1, CudaPieceFloat s2, CudaPieceFloat s3, CudaPieceFloat dis, int batchsize, int m, float margin);
        void Deriv_Dis_Linear(CudaPieceFloat s1deriv, CudaPieceFloat s2deriv, CudaPieceFloat s3deriv, CudaPieceFloat s1, CudaPieceFloat s2, CudaPieceFloat s3, CudaPieceFloat dis, int batchsize, int m, float margin);
        void Deriv_Dis_Rectified(CudaPieceFloat s1deriv, CudaPieceFloat s2deriv, CudaPieceFloat s3deriv, CudaPieceFloat s1, CudaPieceFloat s2, CudaPieceFloat s3, CudaPieceFloat dis, int batchsize, int m, float margin, float eps);
        void Calc_EuclideanDis(CudaPieceFloat s1output, CudaPieceFloat s2output, CudaPieceFloat s3output, CudaPieceFloat dist, int batchsize, int outputLayerSize, float eps);
        void Derive_Cosine_Linear(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps);
        void Derive_Cosine_Rectified(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps);
        void Deriv_Cosine_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps);
        void Derive_Cosine_Linear_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps);
        void Derive_Cosine_Rectified_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps);
        void Matrix_WeightAdd(CudaPieceFloat result, CudaPieceFloat addTerm, int batchsize, int outputLayerSize, CudaPieceFloat mweight, int start, int keep);
        void Matrix_WeightAdd_EX(CudaPieceFloat result, CudaPieceFloat addTerm, CudaPieceInt GPU_Inver_negative_index, CudaPieceInt GPU_Inver_negative_value, int batchsize, int outputLayerSize, CudaPieceFloat mweight, int start, int keep);

        /*----------  Mainly used in backward propagation, computing layer error   --------------*/
        void Deriv_Tanh(CudaPieceFloat errorDeriv, CudaPieceFloat output, int batchsize, int inputDimension);
        void Deriv_Rectified(CudaPieceFloat errorDeriv, CudaPieceFloat output, int batchsize, int inputDimension);

        /*----------  Mainly used in backward propagation, computing weight derivative  --------------*/
        //void SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(BatchSample_Input input_batch, CudaPieceFloat weightDeriv, CudaPieceFloat upperOutputErrorDeriv, int inputDimension, int outputDimension, int winSize);
        void Convolution_Matrix_Product_INTEX(CudaPieceFloat upperOutputErrorDeriv1, CudaPieceInt layerMaxPooling_Index1, CudaPieceFloat upperOutputErrorDeriv2, CudaPieceInt layerMaxPooling_Index2,CudaPieceFloat upperOutputErrorDeriv3, CudaPieceInt layerMaxPooling_Index3, LookupTab wordLT, BatchSample_Input input_batch1, BatchSample_Input input_batch2, BatchSample_Input input_batch3, int winSize, int batchsize, int outputDimension, CudaPieceFloat weightDeriv, int inputDimension);
        void MultiConv_Matrix_Product_INTEX(CudaPieceFloat upperOutputErrorDeriv1, CudaPieceInt layerMaxPooling_Index1, CudaPieceFloat upperOutputErrorDeriv2, CudaPieceInt layerMaxPooling_Index2,CudaPieceFloat upperOutputErrorDeriv3, CudaPieceInt layerMaxPooling_Index3, LookupTab wordLT, BatchSample_Input input_batch1, BatchSample_Input input_batch2, BatchSample_Input input_batch3, int batchsize, int outputDimension, CudaPieceFloat weightDeriv, int inputDimension, int winsize, int fmsize, int accu, int accu_para);
        void MultiConv_Compute_WVDERIV(CudaPieceFloat upperOutputErrorDeriv, CudaPieceInt layerMaxPooling_Index, CudaPieceFloat weight, int batchsize, int outputDimension, CudaPieceFloat inputDeriv, int inputDimension, CudaPieceInt winsizes, CudaPieceInt fmsizes);
        void Conv_Compute_WVDERIV(CudaPieceFloat upperOutputErrorDeriv, CudaPieceInt layerMaxPooling_Index, CudaPieceFloat weight, int batchsize, int outputDimension, CudaPieceFloat inputDeriv, int inputDimension, int winsize);
        void Matrix_Product(CudaPieceFloat lowerOutput1, CudaPieceFloat upperOutputErrorDeriv1, CudaPieceFloat lowerOutput2, CudaPieceFloat upperOutputErrorDeriv2,CudaPieceFloat lowerOutput3, CudaPieceFloat upperOutputErrorDeriv3,CudaPieceFloat weightDeriv, int batchsize, int inputDimension, int outputDimension);
        
        /*----------  Mainly used in backward propagation, computing weight updates and updating weights --------------*/
        void Scale_Matrix(CudaPieceFloat matrix, int inputDimension, int outputDimnsion, float momentum );
        void Matrix_Add(CudaPieceFloat matrix, CudaPieceFloat updates, int inputDimension, int outputDimnsion, float learning_rate);
        void Matrix_Add_REAL(CudaPieceFloat matrix, CudaPieceFloat updates, int inputDimension, int outputDimnsion);

        /*----------  Mainly used in backward propagation in MultiRegression Task --------------*/
        //void Sparse2Dense_Matrix(BatchSample_Input data, CudaPieceFloat matrix, int batchsize, int outputDimension);

        void Zero(CudaPieceFloat matrix, int size);

        void Matrix_Aggragate(CudaPieceFloat a1, CudaPieceFloat a2, CudaPieceFloat a3, CudaPieceFloat b, int batchsize, int m);

        void Cosine_Similarity_EX_Full(CudaPieceFloat a, CudaPieceFloat b, CudaPieceInt neg_list, CudaPieceFloat c, int nTrial, int BATCHSIZE, 
                int batchsize, int dimension, float eps);

        void FillOut_Dist_NCE_Full(CudaPieceFloat dist, CudaPieceInt neg_list, int nTrail, int BATCH_SIZE, int batchsize);

        void Deriv_Cosine_EX_Full(CudaPieceFloat q, CudaPieceFloat d, CudaPieceInt neg_list, CudaPieceFloat dcq,
                CudaPieceFloat dcd, int nTrail, int BATCHSIZE, int batchsize, int m, float eps);

        void Deriv_Cosine_Linear_EX_Full(CudaPieceFloat q, CudaPieceFloat d, CudaPieceInt neg_list, CudaPieceFloat dcq, CudaPieceFloat dcd, 
                int nTrail, int BATCHSIZE, int batchsize, int m, float eps);

        void Deriv_Cosine_Rectified_EX_Full(CudaPieceFloat q, CudaPieceFloat d, CudaPieceInt neg_list, CudaPieceFloat dcq, CudaPieceFloat dcd, 
                int nTrail, int BATCHSIZE, int batchsize, int m, float eps);

        void Matrix_WeightAdd_Full(CudaPieceFloat gpu_floats_a, CudaPieceFloat gpu_floats_b, int nTrail, int BATCHSIZE, int batchsize, int dimension,
                CudaPieceFloat mweight, int start, int keep);

        void Matrix_WeightAdd_EX_Full(CudaPieceFloat gpu_floats_a, CudaPieceFloat gpu_floats_b, CudaPieceInt inver_neg_index,
                CudaPieceInt inver_neg_value, int nTrial, int BATCHSIZE, int batchsize, int dimension, CudaPieceFloat mweight, int start, int keep);

        void Cosine_Similarity_SubSpace(CudaPieceFloat a, CudaPieceFloat b, CudaPieceFloat c, int labelDim, int BATCHSIZE, int batchsize, int subspaceDim, float eps);

        void SoftMax(CudaPieceFloat a, CudaPieceFloat b, int labelDim, int batchsize, float gamma);

        void Deriv_Cosine_Subspace(CudaPieceFloat q, CudaPieceFloat d, CudaPieceFloat dcq, CudaPieceFloat dcd, CudaPieceFloat alpha, int act_type, int batchsize, int labelDim, int subspaceDim, float gamma, float eps);

        void InnerProduct_Similarity(CudaPieceFloat a, CudaPieceFloat b, CudaPieceFloat c, int batchsize, int dimension);

        void Deriv_InnerProduct(CudaPieceFloat q, CudaPieceFloat d, CudaPieceFloat dcq, CudaPieceFloat dcd, CudaPieceFloat alpha, int act_type, int batchsize, int Dim, float gamma, float eps);

        void Matrix_Add_OFFSET(CudaPieceFloat a, int offset_a, CudaPieceFloat b, int offset_b, int len, float mweight);

        void FillOut_Composite(CudaPieceFloat outputLast, BatchSample_Input data, CudaPieceFloat compData, LookupTab contextLT, CudaPieceFloat context, int d1, int d2, int direction);

        void Sparse_Update_Lookup(LookupTab table, LookupTabRunData tableD, int seq1size, int seq2size, int Feature_Dimension, float lr);

        void Sparse_Update_Lookup_Update(CudaPieceFloat tabUpdate, LookupTabRunData tableD, int seq1size, int seq2size, int Feature_Dimension, float lr);
    }

    static class MathOperatorManager
    {
        static IMathOperationManager globalInstance = null;

        public static IMathOperationManager GlobalInstance
        {
            get
            {
                if (globalInstance == null)
                {
                    switch (ParameterSetting.MATH_LIB)
                    {
                        case MathLibType.gpu:
                            globalInstance = new CudaMathOperation();
                            break;
                        case MathLibType.cpu:
                            globalInstance = new BasicMathOperation();
                            break;
                        default:
                            throw new Exception("Error! Unknown Math_LIB " + ParameterSetting.MATH_LIB);
                    }
                }
                return globalInstance;
            }
        }
    }

    class CudaMathOperation : IMathOperationManager
    {

        //public void SEQ_Sparse_Matrix_Multiply_INTEX(BatchSample_Input data, CudaPieceFloat weight, CudaPieceFloat output, int inputDimension, int outputDimension, int winSize)
        //{
        //    Cudalib.SEQ_Sparse_Matrix_Multiply_INTEX(data.Sample_Idx, data.batchsize, data.Seg_Idx, data.Seg_Margin, data.Seg_Len, data.segsize, data.Fea_Idx, data.Fea_Value, data.elementsize,
        //                                weight.CudaPtr, output.CudaPtr, inputDimension, outputDimension, winSize);
                    
        //}


        public void Convolution_Matrix_Multiply_INTEX(BatchSample_Input data, CudaPieceFloat weight, CudaPieceFloat layerPoolingOutput, LookupTab wtab, int inputDimension, int outputDimension, int winSize)
        {
            Cudalib.Convolution_Matrix_Multiply_INTEX(data.Sample_Idx, data.batchsize, data.Word_Idx, data.Seg_Margin, data.elementSize, wtab.LookupTable,
                                        weight.CudaPtr, layerPoolingOutput.CudaPtr, inputDimension, outputDimension, winSize);

        }

        public void MultiConv_Matrix_Multiply_INTEX(BatchSample_Input data, CudaPieceFloat weight, CudaPieceFloat layerPoolingOutput, LookupTab wtab, int inputDimension, int outputDimension, CudaPieceInt wndSizes, CudaPieceInt fmSizes)
        {
            Cudalib.MultiConv_Matrix_Multiply_INTEX(data.Sample_Idx, data.batchsize, data.Word_Idx, data.Seg_Margin, data.elementSize, wtab.LookupTable,
                                        weight.CudaPtr, layerPoolingOutput.CudaPtr, inputDimension, outputDimension, wndSizes.CudaPtr, fmSizes.CudaPtr);

        }

        public void Zero(CudaPieceFloat matrix, int size)
        {
            Cudalib.Zero(matrix.CudaPtr, size);
        }

        public void Max_Pooling(CudaPieceFloat layerPoolingOutput, BatchSample_Input data, CudaPieceFloat output, CudaPieceInt layerMaxPooling_Index, int outputDimension, int winSize)
        {
            Cudalib.Max_Pooling(layerPoolingOutput.CudaPtr, data.Sample_Idx, data.batchsize, output.CudaPtr, layerMaxPooling_Index.CudaPtr, outputDimension, winSize);

        }

        public void Multi_Max_Pooling(CudaPieceFloat layerPoolingOutput, BatchSample_Input data, CudaPieceFloat output, CudaPieceInt layerMaxPooling_Index, int outputDimension, CudaPieceInt wndSizes, CudaPieceInt fmSizes)
        {
            Cudalib.Multi_Max_Pooling(layerPoolingOutput.CudaPtr, data.Sample_Idx, data.batchsize, output.CudaPtr, layerMaxPooling_Index.CudaPtr, outputDimension, wndSizes.CudaPtr, fmSizes.CudaPtr);
        }


        public void Matrix_Multipy(CudaPieceFloat input, CudaPieceFloat weight, CudaPieceFloat output, int batchsize, int inputDimension, int outputDimension, int inverse)
        {
            if (ParameterSetting.CuBlasEnable)
            {
                Cudalib.CUBLAS_Matrix_Multipy(input.CudaPtr, weight.CudaPtr, output.CudaPtr, batchsize, inputDimension, outputDimension, inverse);
            }
            else
            {
                Cudalib.Matrix_Multipy(input.CudaPtr, weight.CudaPtr, output.CudaPtr, batchsize, inputDimension, outputDimension, inverse);
            }
        }


        public void Matrix_Add_Tanh(CudaPieceFloat output, CudaPieceFloat bias, int batchsize, int outputDimension)
        {
            Cudalib.Matrix_Add_Tanh(output.CudaPtr, bias.CudaPtr, batchsize, outputDimension);
        }


        public void Matrix_Add_Vector(CudaPieceFloat output, CudaPieceFloat bias, int batchsize, int outputDimension)
        {
            Cudalib.Matrix_Add_Vector(output.CudaPtr, bias.CudaPtr, batchsize, outputDimension);
        }

        public void Matrix_Rectified_Vector(CudaPieceFloat output, CudaPieceFloat bias, int batchsize, int outputDimension)
        {
            Cudalib.Matrix_Rectified_Vector(output.CudaPtr, bias.CudaPtr, batchsize, outputDimension);
        }

        public void Cosine_Similarity(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int mIndex, int batchsize, int topLayerSize, float eps)
        {
            Cudalib.Cosine_Similarity(srcTopLayerOutput.CudaPtr,
                    tgtTopLayerOutput.CudaPtr, alpha.CudaPtr, nTrailPlus1, BATCH_SIZE, mIndex,
                    batchsize, topLayerSize, eps); // float.Epsilon);
        }

        public void Cosine_Similarity_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int mIndex, int batchsize, int topLayerSize, float eps)
        {
            Cudalib.Cosine_Similarity_EX(srcTopLayerOutput.CudaPtr,
                    tgtTopLayerOutput.CudaPtr, GPU_negative_index.CudaPtr, alpha.CudaPtr, nTrailPlus1, BATCH_SIZE, mIndex,
                    batchsize, topLayerSize, eps); // float.Epsilon);
        }

        public void Calculate_Alpha(CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA)
        {
            Cudalib.Calculate_Alpha(alpha.CudaPtr, nTrailPlus1, BATCH_SIZE, batchsize, GAMMA);
        }

        public void Calculate_Alpha_MXE(CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA)
        {
            Cudalib.Calculate_Alpha_MXE(alpha.CudaPtr, nTrailPlus1, BATCH_SIZE, batchsize, GAMMA);
        }

        public void Calculate_Alpha_NCE(CudaPieceFloat alpha, CudaPieceFloat dist, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA)
        {
            Cudalib.Calculate_Alpha_NCE(alpha.CudaPtr, dist.CudaPtr, nTrailPlus1, BATCH_SIZE, batchsize, GAMMA);
        }

        public void Calculate_Alpha_NCE2(CudaPieceFloat alpha, CudaPieceFloat dist, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA)
        {
            Cudalib.Calculate_Alpha_NCE2(alpha.CudaPtr, dist.CudaPtr, nTrailPlus1, BATCH_SIZE, batchsize, GAMMA);
        }

        public void Calculate_Alpha_PAIRRANK(CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA)
        {
            Cudalib.Calculate_Alpha_PAIRRANK(alpha.CudaPtr, nTrailPlus1, BATCH_SIZE, batchsize, GAMMA);
        }

        public void FillOut_Dist_NCE(CudaPieceFloat dist, CudaPieceInt GPU_negative_index, int nTrailPlus1, int BATCH_SIZE, int mIndex, int batchsize)
        {
            Cudalib.FillOut_Dist_NCE(dist.CudaPtr, GPU_negative_index.CudaPtr, nTrailPlus1, BATCH_SIZE, mIndex, batchsize);
        }

        public void Deriv_Cosine(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            Cudalib.Deriv_Cosine(srcTopLayerOutput.CudaPtr, tgtTopLayerOutput.CudaPtr, srcTopLayerOutputDeriv.CudaPtr, tgtTopLayerOutputDeriv.CudaPtr, batchsize, outputLayerSize, eps);
        }

        public void Deriv_Dis(CudaPieceFloat s1deriv, CudaPieceFloat s2deriv, CudaPieceFloat s3deriv, CudaPieceFloat s1, CudaPieceFloat s2, CudaPieceFloat s3, CudaPieceFloat dis, int batchsize, int m, float margin)
        {
            Cudalib.Deriv_Dis(s1deriv.CudaPtr, s2deriv.CudaPtr, s3deriv.CudaPtr, s1.CudaPtr, s2.CudaPtr, s3.CudaPtr, dis.CudaPtr, batchsize, m, margin);
        }
        public void Deriv_Dis_Linear(CudaPieceFloat s1deriv, CudaPieceFloat s2deriv, CudaPieceFloat s3deriv, CudaPieceFloat s1, CudaPieceFloat s2, CudaPieceFloat s3, CudaPieceFloat dis, int batchsize, int m, float margin)
        {
            Cudalib.Deriv_Dis_Linear(s1deriv.CudaPtr, s2deriv.CudaPtr, s3deriv.CudaPtr, s1.CudaPtr, s2.CudaPtr, s3.CudaPtr, dis.CudaPtr, batchsize, m, margin);
        }
        public void Deriv_Dis_Rectified(CudaPieceFloat s1deriv, CudaPieceFloat s2deriv, CudaPieceFloat s3deriv, CudaPieceFloat s1, CudaPieceFloat s2, CudaPieceFloat s3, CudaPieceFloat dis, int batchsize, int m, float margin, float eps)
        {
            Cudalib.Deriv_Dis_Rectified(s1deriv.CudaPtr, s2deriv.CudaPtr, s3deriv.CudaPtr, s1.CudaPtr, s2.CudaPtr, s3.CudaPtr, dis.CudaPtr, batchsize, m, margin, eps);
        }

        public void Calc_EuclideanDis(CudaPieceFloat s1output, CudaPieceFloat s2output, CudaPieceFloat s3output, CudaPieceFloat dist, int batchsize, int outputLayerSize, float eps)
        {
            Cudalib.Calc_EuclideanDis(s1output.CudaPtr, s2output.CudaPtr, s3output.CudaPtr, dist.CudaPtr, batchsize, outputLayerSize, eps);
        }

        public void Derive_Cosine_Linear(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            Cudalib.Derive_Cosine_Linear(srcTopLayerOutput.CudaPtr, tgtTopLayerOutput.CudaPtr, srcTopLayerOutputDeriv.CudaPtr, tgtTopLayerOutputDeriv.CudaPtr, batchsize, outputLayerSize, eps);
        }

        public void Derive_Cosine_Rectified(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            Cudalib.Derive_Cosine_Rectified(srcTopLayerOutput.CudaPtr, tgtTopLayerOutput.CudaPtr, srcTopLayerOutputDeriv.CudaPtr, tgtTopLayerOutputDeriv.CudaPtr, batchsize, outputLayerSize, eps);
        }

        public void Deriv_Cosine_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            Cudalib.Deriv_Cosine_EX(srcTopLayerOutput.CudaPtr, tgtTopLayerOutput.CudaPtr, GPU_negative_index.CudaPtr, srcTopLayerOutputDeriv.CudaPtr, tgtTopLayerOutputDeriv.CudaPtr, batchsize, outputLayerSize, eps);
        }

        public void Derive_Cosine_Linear_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            Cudalib.Derive_Cosine_Linear_EX(srcTopLayerOutput.CudaPtr, tgtTopLayerOutput.CudaPtr, GPU_negative_index.CudaPtr, srcTopLayerOutputDeriv.CudaPtr, tgtTopLayerOutputDeriv.CudaPtr, batchsize, outputLayerSize, eps);
        }

        public void Derive_Cosine_Rectified_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            Cudalib.Derive_Cosine_Rectified_EX(srcTopLayerOutput.CudaPtr, tgtTopLayerOutput.CudaPtr, GPU_negative_index.CudaPtr, srcTopLayerOutputDeriv.CudaPtr, tgtTopLayerOutputDeriv.CudaPtr, batchsize, outputLayerSize, eps);
        }

        public void Matrix_WeightAdd(CudaPieceFloat result, CudaPieceFloat addTerm, int batchsize, int outputLayerSize, CudaPieceFloat mweight, int start, int keep)
        {
            Cudalib.Matrix_WeightAdd(result.CudaPtr, addTerm.CudaPtr, batchsize, outputLayerSize, mweight.CudaPtr, start, keep);
        }

        public void Matrix_WeightAdd_EX(CudaPieceFloat result, CudaPieceFloat addTerm, CudaPieceInt GPU_Inver_negative_index, CudaPieceInt GPU_Inver_negative_value, int batchsize, int outputLayerSize, CudaPieceFloat mweight, int start, int keep)
        {
            Cudalib.Matrix_WeightAdd_EX(result.CudaPtr, addTerm.CudaPtr, GPU_Inver_negative_index.CudaPtr, GPU_Inver_negative_value.CudaPtr, batchsize, outputLayerSize, mweight.CudaPtr, start, keep);
        }

        public void Deriv_Tanh(CudaPieceFloat errorDeriv, CudaPieceFloat output, int batchsize, int inputDimension)
        {
            Cudalib.Deriv_Tanh(errorDeriv.CudaPtr, output.CudaPtr, batchsize, inputDimension);
        }

        public void Deriv_Rectified(CudaPieceFloat errorDeriv, CudaPieceFloat output, int batchsize, int inputDimension)
        {
            Cudalib.Deriv_Rectified(errorDeriv.CudaPtr, output.CudaPtr, batchsize, inputDimension);
        }


        //public void SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(BatchSample_Input input_batch, CudaPieceFloat weightDeriv, CudaPieceFloat upperOutputErrorDeriv, int inputDimension, int outputDimension, int winSize)
        //{
        //    Cudalib.SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(input_batch.Sample_Idx, input_batch.batchsize, input_batch.Seg_Idx, input_batch.Seg_Margin, input_batch.Seg_Len, input_batch.segsize, input_batch.Fea_Idx, input_batch.Fea_Value, input_batch.elementsize,
        //                       weightDeriv.CudaPtr, upperOutputErrorDeriv.CudaPtr, inputDimension, outputDimension, winSize);
        //}


        public void Convolution_Matrix_Product_INTEX(CudaPieceFloat upperOutputErrorDeriv1, CudaPieceInt layerMaxPooling_Index1, CudaPieceFloat upperOutputErrorDeriv2, CudaPieceInt layerMaxPooling_Index2,CudaPieceFloat upperOutputErrorDeriv3, CudaPieceInt layerMaxPooling_Index3, LookupTab wordLT, BatchSample_Input input_batch1, BatchSample_Input input_batch2, BatchSample_Input input_batch3, int winSize, int batchsize, int outputDimension, CudaPieceFloat weightDeriv, int inputDimension)
        {
            Cudalib.Convolution_Matrix_Product_INTEX(upperOutputErrorDeriv1.CudaPtr, layerMaxPooling_Index1.CudaPtr, upperOutputErrorDeriv2.CudaPtr, layerMaxPooling_Index2.CudaPtr, upperOutputErrorDeriv3.CudaPtr, layerMaxPooling_Index3.CudaPtr, wordLT.LookupTable, input_batch1.Word_Idx, input_batch2.Word_Idx, input_batch3.Word_Idx, winSize,
                                     batchsize, outputDimension, weightDeriv.CudaPtr, inputDimension);
        }

        public void MultiConv_Matrix_Product_INTEX(CudaPieceFloat upperOutputErrorDeriv1, CudaPieceInt layerMaxPooling_Index1, CudaPieceFloat upperOutputErrorDeriv2, CudaPieceInt layerMaxPooling_Index2,CudaPieceFloat upperOutputErrorDeriv3, CudaPieceInt layerMaxPooling_Index3, LookupTab wordLT, BatchSample_Input input_batch1, BatchSample_Input input_batch2, BatchSample_Input input_batch3, int batchsize, int outputDimension, CudaPieceFloat weightDeriv, int inputDimension, int winsize, int fmsize, int accu, int accu_para)
        {
            Cudalib.MultiConv_Matrix_Product_INTEX(upperOutputErrorDeriv1.CudaPtr, layerMaxPooling_Index1.CudaPtr, upperOutputErrorDeriv2.CudaPtr, layerMaxPooling_Index2.CudaPtr, upperOutputErrorDeriv3.CudaPtr, layerMaxPooling_Index3.CudaPtr, wordLT.LookupTable, input_batch1.Word_Idx, input_batch2.Word_Idx, input_batch3.Word_Idx,
                                    batchsize, outputDimension, weightDeriv.CudaPtr, inputDimension, winsize, fmsize, accu, accu_para);
        }

        public void MultiConv_Compute_WVDERIV(CudaPieceFloat upperOutputErrorDeriv, CudaPieceInt layerMaxPooling_Index, CudaPieceFloat weight, int batchsize, int outputDimension, CudaPieceFloat inputDeriv, int inputDimension, CudaPieceInt winsizes, CudaPieceInt fmsizes)
        {
            Cudalib.MultiConv_Compute_WVDERIV(upperOutputErrorDeriv.CudaPtr, layerMaxPooling_Index.CudaPtr, weight.CudaPtr, batchsize, outputDimension, inputDeriv.CudaPtr, inputDimension, winsizes.CudaPtr, fmsizes.CudaPtr);
        }

        public void Conv_Compute_WVDERIV(CudaPieceFloat upperOutputErrorDeriv, CudaPieceInt layerMaxPooling_Index, CudaPieceFloat weight, int batchsize, int outputDimension, CudaPieceFloat inputDeriv, int inputDimension, int winsize)
        {
            Cudalib.Conv_Compute_WVDERIV(upperOutputErrorDeriv.CudaPtr, layerMaxPooling_Index.CudaPtr, weight.CudaPtr, batchsize, outputDimension, inputDeriv.CudaPtr, inputDimension, winsize);
        }


        public void Matrix_Product(CudaPieceFloat lowerOutput1, CudaPieceFloat upperOutputErrorDeriv1, CudaPieceFloat lowerOutput2, CudaPieceFloat upperOutputErrorDeriv2,CudaPieceFloat lowerOutput3, CudaPieceFloat upperOutputErrorDeriv3, CudaPieceFloat weightDeriv, int batchsize, int inputDimension, int outputDimension)
        {
            Cudalib.Matrix_Product(lowerOutput1.CudaPtr, upperOutputErrorDeriv1.CudaPtr, lowerOutput2.CudaPtr, upperOutputErrorDeriv2.CudaPtr,lowerOutput3.CudaPtr, upperOutputErrorDeriv3.CudaPtr,weightDeriv.CudaPtr,
                        batchsize, inputDimension, outputDimension);
        }

        public void Matrix_Aggragate(CudaPieceFloat a1, CudaPieceFloat a2, CudaPieceFloat a3, CudaPieceFloat b, int batchsize, int m)
        {
            Cudalib.Matrix_Aggragate(a1.CudaPtr, a2.CudaPtr, a3.CudaPtr, b.CudaPtr, batchsize, m);
        }

        public void Scale_Matrix(CudaPieceFloat matrix, int inputDimension, int outputDimnsion, float momentum)
        {
            Cudalib.Scale_Matrix(matrix.CudaPtr, inputDimension, outputDimnsion, momentum);
        }

        public void Matrix_Add(CudaPieceFloat matrix, CudaPieceFloat updates, int inputDimension, int outputDimnsion, float learning_rate)
        {
            Cudalib.Matrix_Add(matrix.CudaPtr, updates.CudaPtr, inputDimension, outputDimnsion, learning_rate);
        }

        public void Matrix_Add_REAL(CudaPieceFloat matrix, CudaPieceFloat updates, int inputDimension, int outputDimnsion)
        {
            Cudalib.Matrix_Add_REAL(matrix.CudaPtr, updates.CudaPtr, inputDimension, outputDimnsion);
        }

        //public void Sparse2Dense_Matrix(BatchSample_Input data, CudaPieceFloat matrix, int batchsize, int outputDimension)
        //{
        //    Cudalib.Sparse2Dense_Matrix(data.Seg_Idx, data.Fea_Idx, data.Fea_Value, matrix.CudaPtr, batchsize, outputDimension);
        //}


        public void Cosine_Similarity_EX_Full(CudaPieceFloat a, CudaPieceFloat b, CudaPieceInt neg_list, CudaPieceFloat c, int nTrial, int BATCHSIZE,
                int batchsize, int dimension, float eps)
        {
            Cudalib.Cosine_Similarity_EX_Full(a.CudaPtr, b.CudaPtr, neg_list.CudaPtr, c.CudaPtr, nTrial, BATCHSIZE, batchsize, dimension, eps);
        }

        public void FillOut_Dist_NCE_Full(CudaPieceFloat dist, CudaPieceInt neg_list, int nTrail, int BATCH_SIZE, int batchsize)
        {
            Cudalib.FillOut_Dist_NCE_Full(dist.CudaPtr, neg_list.CudaPtr, nTrail, BATCH_SIZE, batchsize);
        }

        public void Deriv_Cosine_EX_Full(CudaPieceFloat q, CudaPieceFloat d, CudaPieceInt neg_list, CudaPieceFloat dcq,
                CudaPieceFloat dcd, int nTrail, int BATCHSIZE, int batchsize, int m, float eps)
        {
            Cudalib.Deriv_Cosine_EX_Full(q.CudaPtr, d.CudaPtr, neg_list.CudaPtr, dcq.CudaPtr, dcd.CudaPtr, nTrail, BATCHSIZE, batchsize, m, eps);
        }

        public void Deriv_Cosine_Linear_EX_Full(CudaPieceFloat q, CudaPieceFloat d, CudaPieceInt neg_list, CudaPieceFloat dcq, CudaPieceFloat dcd,
                int nTrail, int BATCHSIZE, int batchsize, int m, float eps)
        {
            Cudalib.Deriv_Cosine_Linear_EX_Full(q.CudaPtr, d.CudaPtr, neg_list.CudaPtr, dcq.CudaPtr, dcd.CudaPtr, nTrail, BATCHSIZE, batchsize, m, eps);
        }

        public void Deriv_Cosine_Rectified_EX_Full(CudaPieceFloat q, CudaPieceFloat d, CudaPieceInt neg_list, CudaPieceFloat dcq, CudaPieceFloat dcd,
                int nTrail, int BATCHSIZE, int batchsize, int m, float eps)
        {
            Cudalib.Deriv_Cosine_Rectified_EX_Full(q.CudaPtr, d.CudaPtr, neg_list.CudaPtr, dcq.CudaPtr, dcd.CudaPtr, nTrail, BATCHSIZE, batchsize, m, eps);
        }

        public void Matrix_WeightAdd_Full(CudaPieceFloat gpu_floats_a, CudaPieceFloat gpu_floats_b, int nTrail, int BATCHSIZE, int batchsize, int dimension,
                CudaPieceFloat mweight, int start, int keep)
        {
            Cudalib.Matrix_WeightAdd_Full(gpu_floats_a.CudaPtr, gpu_floats_b.CudaPtr, nTrail, BATCHSIZE, batchsize, dimension,
                mweight.CudaPtr, start, keep);
        }

        public void Matrix_WeightAdd_EX_Full(CudaPieceFloat gpu_floats_a, CudaPieceFloat gpu_floats_b, CudaPieceInt inver_neg_index,
                CudaPieceInt inver_neg_value, int nTrial, int BATCHSIZE, int batchsize, int dimension, CudaPieceFloat mweight,
                int start, int keep)
        {
            Cudalib.Matrix_WeightAdd_EX_Full(gpu_floats_a.CudaPtr, gpu_floats_b.CudaPtr, inver_neg_index.CudaPtr,
                inver_neg_value.CudaPtr, nTrial, BATCHSIZE, batchsize, dimension, mweight.CudaPtr, start, keep);
        }

        public void Cosine_Similarity_SubSpace(CudaPieceFloat a, CudaPieceFloat b, CudaPieceFloat c, int labelDim, int BATCHSIZE, int batchsize, int subspaceDim, float eps)
        {
            Cudalib.Cosine_Similarity_SubSpace(a.CudaPtr, b.CudaPtr, c.CudaPtr, labelDim, BATCHSIZE, batchsize, subspaceDim, eps);
        }

        public void SoftMax(CudaPieceFloat a, CudaPieceFloat b, int labelDim, int batchsize, float gamma)
        {
            Cudalib.SoftMax(a.CudaPtr, b.CudaPtr, labelDim, batchsize, gamma);
        }

        public void Deriv_Cosine_Subspace(CudaPieceFloat q, CudaPieceFloat d, CudaPieceFloat dcq, CudaPieceFloat dcd, CudaPieceFloat alpha, int act_type, int batchsize, int labelDim, int subspaceDim, float gamma, float eps)
        {
            Cudalib.Deriv_Cosine_Subspace(q.CudaPtr, d.CudaPtr, dcq.CudaPtr, dcd.CudaPtr, alpha.CudaPtr, act_type, batchsize, labelDim, subspaceDim, gamma, eps);
        }

        public void InnerProduct_Similarity(CudaPieceFloat a, CudaPieceFloat b, CudaPieceFloat c, int batchsize, int dimension)
        {
            Cudalib.InnerProduct_Similarity(a.CudaPtr, b.CudaPtr, c.CudaPtr, batchsize, dimension);
        }

        public void Deriv_InnerProduct(CudaPieceFloat q, CudaPieceFloat d, CudaPieceFloat dcq, CudaPieceFloat dcd, CudaPieceFloat alpha, int act_type, int batchsize, int Dim, float gamma, float eps)
        {
            Cudalib.Deriv_InnerProduct(q.CudaPtr, d.CudaPtr, dcq.CudaPtr, dcd.CudaPtr, alpha.CudaPtr, act_type, batchsize, Dim, gamma, eps);
        }

        public void Matrix_Add_OFFSET(CudaPieceFloat a, int offset_a, CudaPieceFloat b, int offset_b, int len, float mweight)
        {
            Cudalib.Matrix_Add_OFFSET(a.CudaPtr, offset_a, b.CudaPtr, offset_b, len, mweight);
        }

        public void FillOut_Composite(CudaPieceFloat outputLast, BatchSample_Input data, CudaPieceFloat compData, LookupTab contextLT, CudaPieceFloat context, int d1, int d2, int direction)
        {
            if (direction != 0)
                Cudalib.FillOut_Composite(outputLast.CudaPtr, data.Fea_Idx, compData.CudaPtr, contextLT.LookupTable, d1, d2, data.batchsize, direction);
            else
                Cudalib.FillOut_Composite(outputLast.CudaPtr, data.Fea_Idx, compData.CudaPtr, context.CudaPtr, d1, d2, data.batchsize, direction);
        }

        public void Sparse_Update_Lookup(LookupTab table, LookupTabRunData tableD, int seq1size, int seq2size, int Feature_Dimension, float lr)
        {
            Cudalib.Sparse_Update_Lookup(table.LookupTable, tableD.uniqueWordID.CudaPtr, tableD.uniqueWordIdx.CudaPtr, tableD.Sequence.CudaPtr, tableD.InputDeriv[0].CudaPtr, tableD.InputDeriv[1].CudaPtr, tableD.InputDeriv[2].CudaPtr, seq1size, seq2size, tableD.uniqueNum, Feature_Dimension, lr);
        }

        public void Sparse_Update_Lookup_Update(CudaPieceFloat tabUpdate, LookupTabRunData tableD, int seq1size, int seq2size, int Feature_Dimension, float lr)
        {
            Cudalib.Sparse_Update_Lookup_Update(tabUpdate.CudaPtr, tableD.uniqueWordID.CudaPtr, tableD.uniqueWordIdx.CudaPtr, tableD.Sequence.CudaPtr, tableD.InputDeriv[0].CudaPtr, tableD.InputDeriv[1].CudaPtr, tableD.InputDeriv[2].CudaPtr, seq1size, seq2size, tableD.uniqueNum, Feature_Dimension, lr);
        }
    }

    class BasicMathOperation : IMathOperationManager
    {
        public void Cosine_Similarity_EX_Full(CudaPieceFloat a, CudaPieceFloat b, CudaPieceInt neg_list, CudaPieceFloat c, int nTrial, int BATCHSIZE,
                int batchsize, int dimension, float eps)
        {
            BasicMathlib.Cosine_Similarity_EX_Full(a.MemPtr, b.MemPtr, neg_list.MemPtr, c.MemPtr, nTrial, BATCHSIZE, batchsize, dimension, eps);
        }

        public void FillOut_Dist_NCE_Full(CudaPieceFloat dist, CudaPieceInt neg_list, int nTrail, int BATCH_SIZE, int batchsize)
        {
            BasicMathlib.FillOut_Dist_NCE_Full(dist.MemPtr, neg_list.MemPtr, nTrail, BATCH_SIZE, batchsize);
        }

        public void Deriv_Cosine_EX_Full(CudaPieceFloat q, CudaPieceFloat d, CudaPieceInt neg_list, CudaPieceFloat dcq,
                CudaPieceFloat dcd, int nTrail, int BATCHSIZE, int batchsize, int m, float eps)
        {
            BasicMathlib.Deriv_Cosine_EX_Full(q.MemPtr, d.MemPtr, neg_list.MemPtr, dcq.MemPtr, dcd.MemPtr, nTrail, BATCHSIZE, batchsize, m, eps);
        }

        public void Deriv_Cosine_Linear_EX_Full(CudaPieceFloat q, CudaPieceFloat d, CudaPieceInt neg_list, CudaPieceFloat dcq, CudaPieceFloat dcd,
                int nTrail, int BATCHSIZE, int batchsize, int m, float eps)
        {
            BasicMathlib.Deriv_Cosine_Linear_EX_Full(q.MemPtr, d.MemPtr, neg_list.MemPtr, dcq.MemPtr, dcd.MemPtr, nTrail, BATCHSIZE, batchsize, m, eps);
        }

        public void Deriv_Cosine_Rectified_EX_Full(CudaPieceFloat q, CudaPieceFloat d, CudaPieceInt neg_list, CudaPieceFloat dcq, CudaPieceFloat dcd,
                int nTrail, int BATCHSIZE, int batchsize, int m, float eps)
        {
            BasicMathlib.Deriv_Cosine_Rectified_EX_Full(q.MemPtr, d.MemPtr, neg_list.MemPtr, dcq.MemPtr, dcd.MemPtr, nTrail, BATCHSIZE, batchsize, m, eps);
        }

        public void Matrix_WeightAdd_Full(CudaPieceFloat gpu_floats_a, CudaPieceFloat gpu_floats_b, int nTrail, int BATCHSIZE, int batchsize, int dimension,
                CudaPieceFloat mweight, int start, int keep)
        {
            BasicMathlib.Matrix_WeightAdd_Full(gpu_floats_a.MemPtr, gpu_floats_b.MemPtr, nTrail, BATCHSIZE, batchsize, dimension,
                mweight.MemPtr, start, keep);
        }

        public void Matrix_WeightAdd_EX_Full(CudaPieceFloat gpu_floats_a, CudaPieceFloat gpu_floats_b, CudaPieceInt inver_neg_index,
                CudaPieceInt inver_neg_value, int nTrial, int BATCHSIZE, int batchsize, int dimension, CudaPieceFloat mweight,
                int start, int keep)
        {
            BasicMathlib.Matrix_WeightAdd_EX_Full(gpu_floats_a.MemPtr, gpu_floats_b.MemPtr, inver_neg_index.MemPtr,
                inver_neg_value.MemPtr, nTrial, BATCHSIZE, batchsize, dimension, mweight.MemPtr, start, keep);
        }

        //public void SEQ_Sparse_Matrix_Multiply_INTEX(BatchSample_Input data, CudaPieceFloat weight, CudaPieceFloat output, int inputDimension, int outputDimension, int winSize)
        //{
        //    BasicMathlib.SEQ_Sparse_Matrix_Multiply_INTEX(data.Sample_Idx_Mem, data.batchsize, data.Seg_Idx_Mem, data.Seg_Margin_Mem, data.Seg_Len_Mem, data.segsize, data.Fea_Idx_Mem, data.Fea_Value_Mem, data.elementsize,
        //                                weight.MemPtr, output.MemPtr, inputDimension, outputDimension, winSize);
        //}

        public void Convolution_Matrix_Multiply_INTEX(BatchSample_Input data, CudaPieceFloat weight, CudaPieceFloat layerPoolingOutput, LookupTab wtab, int inputDimension, int outputDimension, int winSize)
        {
            // not implemented
            //BasicMathlib.Convolution_Sparse_Matrix_Multiply_INTEX(data.Sample_Idx_Mem, data.batchsize, data.Seg_Idx_Mem, data.Seg_Margin_Mem, data.Seg_Len_Mem, data.segsize, data.Fea_Idx_Mem, data.Fea_Value_Mem, data.elementsize,
            //                            weight.MemPtr, layerPoolingOutput.MemPtr, inputDimension, outputDimension, winSize);

        }

        public void MultiConv_Matrix_Multiply_INTEX(BatchSample_Input data, CudaPieceFloat weight, CudaPieceFloat layerPoolingOutput, LookupTab wtab, int inputDimension, int outputDimension, CudaPieceInt wndSizes, CudaPieceInt fmSizes)
        {
            // not implemented

        }

        public void Max_Pooling(CudaPieceFloat layerPoolingOutput, BatchSample_Input data, CudaPieceFloat output, CudaPieceInt layerMaxPooling_Index, int outputDimension, int winSize)
        {
            // not implemented
            //BasicMathlib.Max_Pooling(layerPoolingOutput.MemPtr, data.Sample_Idx_Mem, data.batchsize, output.MemPtr, layerMaxPooling_Index.MemPtr, outputDimension);
        }

        public void Multi_Max_Pooling(CudaPieceFloat layerPoolingOutput, BatchSample_Input data, CudaPieceFloat output, CudaPieceInt layerMaxPooling_Index, int outputDimension, CudaPieceInt wndSizes, CudaPieceInt fmSizes)
        {
            // not implemented
        }

        public void Matrix_Multipy(CudaPieceFloat input, CudaPieceFloat weight, CudaPieceFloat output, int batchsize, int inputDimension, int outputDimension, int inverse)
        {
            BasicMathlib.Matrix_Multipy(input.MemPtr, weight.MemPtr, output.MemPtr, batchsize, inputDimension, outputDimension, inverse);
        }

        public void Matrix_Add_Tanh(CudaPieceFloat output, CudaPieceFloat bias, int batchsize, int outputDimension)
        {
            BasicMathlib.Matrix_Add_Tanh(output.MemPtr, bias.MemPtr, batchsize, outputDimension);
        }

        public void Matrix_Add_Vector(CudaPieceFloat output, CudaPieceFloat bias, int batchsize, int outputDimension)
        {
            BasicMathlib.Matrix_Add_Vector(output.MemPtr, bias.MemPtr, batchsize, outputDimension);
        }

        public void Matrix_Rectified_Vector(CudaPieceFloat output, CudaPieceFloat bias, int batchsize, int outputDimension)
        {
            BasicMathlib.Matrix_Rectified_Vector(output.MemPtr, bias.MemPtr, batchsize, outputDimension);
        }

        public void Cosine_Similarity(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int mIndex, int batchsize, int topLayerSize, float eps)
        {
            BasicMathlib.Cosine_Similarity(srcTopLayerOutput.MemPtr,
                    tgtTopLayerOutput.MemPtr, alpha.MemPtr, nTrailPlus1, BATCH_SIZE, mIndex,
                    batchsize, topLayerSize, eps);
        }

        public void Cosine_Similarity_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int mIndex, int batchsize, int topLayerSize, float eps)
        {
            BasicMathlib.Cosine_Similarity_EX(srcTopLayerOutput.MemPtr,
                    tgtTopLayerOutput.MemPtr, GPU_negative_index.MemPtr, alpha.MemPtr, nTrailPlus1, BATCH_SIZE, mIndex,
                    batchsize, topLayerSize, eps);
        }

        public void Calculate_Alpha(CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA)
        {
            BasicMathlib.Calculate_Alpha(alpha.MemPtr, nTrailPlus1, BATCH_SIZE, batchsize, GAMMA);
        }

        public void Calculate_Alpha_MXE(CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA)
        {
            BasicMathlib.Calculate_Alpha_MXE(alpha.MemPtr, nTrailPlus1, BATCH_SIZE, batchsize, GAMMA);
        }

        public void Calculate_Alpha_NCE(CudaPieceFloat alpha, CudaPieceFloat dist, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA)
        {
            BasicMathlib.Calculate_Alpha_NCE(alpha.MemPtr, dist.MemPtr, nTrailPlus1, BATCH_SIZE, batchsize, GAMMA);
        }

        public void Calculate_Alpha_NCE2(CudaPieceFloat alpha, CudaPieceFloat dist, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA)
        {
            BasicMathlib.Calculate_Alpha_NCE2(alpha.MemPtr, dist.MemPtr, nTrailPlus1, BATCH_SIZE, batchsize, GAMMA);
        }

        public void Calculate_Alpha_PAIRRANK(CudaPieceFloat alpha, int nTrailPlus1, int BATCH_SIZE, int batchsize, float GAMMA)
        {
            BasicMathlib.Calculate_Alpha_PAIRRANK(alpha.MemPtr, nTrailPlus1, BATCH_SIZE, batchsize, GAMMA);
        }

        public void FillOut_Dist_NCE(CudaPieceFloat dist, CudaPieceInt GPU_negative_index, int nTrailPlus1, int BATCH_SIZE, int mIndex, int batchsize)
        {
            BasicMathlib.FillOut_Dist_NCE(dist.MemPtr, GPU_negative_index.MemPtr, nTrailPlus1, BATCH_SIZE, mIndex, batchsize);
        }

        public void Deriv_Cosine(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            BasicMathlib.Deriv_Cosine(srcTopLayerOutput.MemPtr, tgtTopLayerOutput.MemPtr, srcTopLayerOutputDeriv.MemPtr, tgtTopLayerOutputDeriv.MemPtr, batchsize, outputLayerSize, eps);
        }

        public void Deriv_Dis(CudaPieceFloat s1deriv, CudaPieceFloat s2deriv, CudaPieceFloat s3deriv, CudaPieceFloat s1, CudaPieceFloat s2, CudaPieceFloat s3, CudaPieceFloat dis, int batchsize, int m, float margin)
        {
            //Cudalib.Deriv_Dis(s1deriv.CudaPtr, s2deriv.CudaPtr, s3deriv.CudaPtr, s1.CudaPtr, s2.CudaPtr, s3.CudaPtr, dis.CudaPtr, batchsize, m);
        }
        public void Deriv_Dis_Linear(CudaPieceFloat s1deriv, CudaPieceFloat s2deriv, CudaPieceFloat s3deriv, CudaPieceFloat s1, CudaPieceFloat s2, CudaPieceFloat s3, CudaPieceFloat dis, int batchsize, int m, float margin)
        {
            //Cudalib.Deriv_Dis_Linear(s1deriv.CudaPtr, s2deriv.CudaPtr, s3deriv.CudaPtr, s1.CudaPtr, s2.CudaPtr, s3.CudaPtr, dis.CudaPtr, batchsize, m);
        }
        public void Deriv_Dis_Rectified(CudaPieceFloat s1deriv, CudaPieceFloat s2deriv, CudaPieceFloat s3deriv, CudaPieceFloat s1, CudaPieceFloat s2, CudaPieceFloat s3, CudaPieceFloat dis, int batchsize, int m, float margin, float eps)
        {
            //Cudalib.Deriv_Dis_Rectified(s1deriv.CudaPtr, s2deriv.CudaPtr, s3deriv.CudaPtr, s1.CudaPtr, s2.CudaPtr, s3.CudaPtr, dis.CudaPtr, batchsize, m, eps);
        }

        public void Calc_EuclideanDis(CudaPieceFloat s1output, CudaPieceFloat s2output, CudaPieceFloat s3output, CudaPieceFloat dist, int batchsize, int outputLayerSize, float eps)
        {
            //not implemented
        }

        public void Derive_Cosine_Linear(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            BasicMathlib.Derive_Cosine_Linear(srcTopLayerOutput.MemPtr, tgtTopLayerOutput.MemPtr, srcTopLayerOutputDeriv.MemPtr, tgtTopLayerOutputDeriv.MemPtr, batchsize, outputLayerSize, eps);
        }

        public void Derive_Cosine_Rectified(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            BasicMathlib.Derive_Cosine_Rectified(srcTopLayerOutput.MemPtr, tgtTopLayerOutput.MemPtr, srcTopLayerOutputDeriv.MemPtr, tgtTopLayerOutputDeriv.MemPtr, batchsize, outputLayerSize, eps);
        }

        public void Deriv_Cosine_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            BasicMathlib.Deriv_Cosine_EX(srcTopLayerOutput.MemPtr, tgtTopLayerOutput.MemPtr, GPU_negative_index.MemPtr, srcTopLayerOutputDeriv.MemPtr, tgtTopLayerOutputDeriv.MemPtr, batchsize, outputLayerSize, eps);
        }

        public void Derive_Cosine_Linear_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            BasicMathlib.Derive_Cosine_Linear_EX(srcTopLayerOutput.MemPtr, tgtTopLayerOutput.MemPtr, GPU_negative_index.MemPtr, srcTopLayerOutputDeriv.MemPtr, tgtTopLayerOutputDeriv.MemPtr, batchsize, outputLayerSize, eps);
        }

        public void Derive_Cosine_Rectified_EX(CudaPieceFloat srcTopLayerOutput, CudaPieceFloat tgtTopLayerOutput, CudaPieceInt GPU_negative_index, CudaPieceFloat srcTopLayerOutputDeriv, CudaPieceFloat tgtTopLayerOutputDeriv, int batchsize, int outputLayerSize, float eps)
        {
            BasicMathlib.Derive_Cosine_Rectified_EX(srcTopLayerOutput.MemPtr, tgtTopLayerOutput.MemPtr, GPU_negative_index.MemPtr, srcTopLayerOutputDeriv.MemPtr, tgtTopLayerOutputDeriv.MemPtr, batchsize, outputLayerSize, eps);
        }

        public void Matrix_WeightAdd(CudaPieceFloat result, CudaPieceFloat addTerm, int batchsize, int outputLayerSize, CudaPieceFloat mweight, int start, int keep)
        {
            BasicMathlib.Matrix_WeightAdd(result.MemPtr, addTerm.MemPtr, batchsize, outputLayerSize, mweight.MemPtr, start, keep);
        }

        public void Matrix_WeightAdd_EX(CudaPieceFloat result, CudaPieceFloat addTerm, CudaPieceInt GPU_Inver_negative_index, CudaPieceInt GPU_Inver_negative_value, int batchsize, int outputLayerSize, CudaPieceFloat mweight, int start, int keep)
        {
            BasicMathlib.Matrix_WeightAdd_EX(result.MemPtr, addTerm.MemPtr, GPU_Inver_negative_index.MemPtr, GPU_Inver_negative_value.MemPtr, batchsize, outputLayerSize, mweight.MemPtr, start, keep);
        }

        public void Deriv_Tanh(CudaPieceFloat errorDeriv, CudaPieceFloat output, int batchsize, int inputDimension)
        {
            BasicMathlib.Deriv_Tanh(errorDeriv.MemPtr, output.MemPtr, batchsize, inputDimension);
        }

        public void Deriv_Rectified(CudaPieceFloat errorDeriv, CudaPieceFloat output, int batchsize, int inputDimension)
        {
            BasicMathlib.Deriv_Rectified(errorDeriv.MemPtr, output.MemPtr, batchsize, inputDimension);
        }

        //public void SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(BatchSample_Input input_batch, CudaPieceFloat weightDeriv, CudaPieceFloat upperOutputErrorDeriv, int inputDimension, int outputDimension, int winSize)
        //{
        //    BasicMathlib.SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(input_batch.Sample_Idx_Mem, input_batch.batchsize, input_batch.Seg_Idx_Mem, input_batch.Seg_Margin_Mem, input_batch.Seg_Len_Mem, input_batch.segsize, input_batch.Fea_Idx_Mem, input_batch.Fea_Value_Mem, input_batch.elementsize,
        //                       weightDeriv.MemPtr, upperOutputErrorDeriv.MemPtr, inputDimension, outputDimension, winSize);
        //}

        public void Convolution_Matrix_Product_INTEX(CudaPieceFloat upperOutputErrorDeriv1, CudaPieceInt layerMaxPooling_Index1, CudaPieceFloat upperOutputErrorDeriv2, CudaPieceInt layerMaxPooling_Index2,CudaPieceFloat upperOutputErrorDeriv3, CudaPieceInt layerMaxPooling_Index3, LookupTab wordLT, BatchSample_Input input_batch1, BatchSample_Input input_batch2, BatchSample_Input input_batch3, int winSize, int batchsize, int outputDimension, CudaPieceFloat weightDeriv, int inputDimension)
        {
            // not implemented
            //BasicMathlib.Convolution_Sparse_Matrix_Product_INTEX(upperOutputErrorDeriv.MemPtr, layerMaxPooling_Index.MemPtr, input_batch.Seg_Idx_Mem, input_batch.Seg_Margin_Mem, input_batch.segsize, winSize,
            //                         batchsize, outputDimension, input_batch.Fea_Idx_Mem, input_batch.Fea_Value_Mem, weightDeriv.MemPtr, inputDimension);
        }

        public void MultiConv_Matrix_Product_INTEX(CudaPieceFloat upperOutputErrorDeriv1, CudaPieceInt layerMaxPooling_Index1, CudaPieceFloat upperOutputErrorDeriv2, CudaPieceInt layerMaxPooling_Index2,CudaPieceFloat upperOutputErrorDeriv3, CudaPieceInt layerMaxPooling_Index3, LookupTab wordLT, BatchSample_Input input_batch1, BatchSample_Input input_batch2, BatchSample_Input input_batch3, int batchsize, int outputDimension, CudaPieceFloat weightDeriv, int inputDimension, int winsize, int fmsize, int accu, int accu_para)
        {
            //not implemented
        }

        public void MultiConv_Compute_WVDERIV(CudaPieceFloat upperOutputErrorDeriv, CudaPieceInt layerMaxPooling_Index, CudaPieceFloat weight, int batchsize, int outputDimension, CudaPieceFloat inputDeriv, int inputDimension, CudaPieceInt winsizes, CudaPieceInt fmsizes)
        {
            //not implemented
        }

        public void Conv_Compute_WVDERIV(CudaPieceFloat upperOutputErrorDeriv, CudaPieceInt layerMaxPooling_Index, CudaPieceFloat weight, int batchsize, int outputDimension, CudaPieceFloat inputDeriv, int inputDimension, int winsize)
        {
            //not implemented
        }

        public void Matrix_Product(CudaPieceFloat lowerOutput1, CudaPieceFloat upperOutputErrorDeriv1, CudaPieceFloat lowerOutput2, CudaPieceFloat upperOutputErrorDeriv2,CudaPieceFloat lowerOutput3, CudaPieceFloat upperOutputErrorDeriv3, CudaPieceFloat weightDeriv, int batchsize, int inputDimension, int outputDimension)
        {
            // not implemented
            //BasicMathlib.Matrix_Product(lowerOutput.MemPtr, upperOutputErrorDeriv.MemPtr, weightDeriv.MemPtr,
            //            batchsize, inputDimension, outputDimension);

        }


        public void Scale_Matrix(CudaPieceFloat matrix, int inputDimension, int outputDimnsion, float momentum)
        {
            BasicMathlib.Scale_Matrix(matrix.MemPtr, inputDimension, outputDimnsion, momentum);
        }

        public void Matrix_Add(CudaPieceFloat matrix, CudaPieceFloat updates, int inputDimension, int outputDimnsion, float learning_rate)
        {
            BasicMathlib.Matrix_Add(matrix.MemPtr, updates.MemPtr, inputDimension, outputDimnsion, learning_rate);
        }

        public void Matrix_Add_REAL(CudaPieceFloat matrix, CudaPieceFloat updates, int inputDimension, int outputDimnsion)
        {
            // not implemented
        }

        //public void Sparse2Dense_Matrix(BatchSample_Input data, CudaPieceFloat matrix, int batchsize, int outputDimension)
        //{
        //    BasicMathlib.Sparse2Dense_Matrix(data.Seg_Idx_Mem, data.Fea_Idx_Mem, data.Fea_Value_Mem, matrix.MemPtr, batchsize, outputDimension);
        //}

        public void Zero(CudaPieceFloat matrix, int size)
        {
            Array.Clear(matrix.MemPtr, 0, matrix.MemPtr.Length);
        }
        
        public void Matrix_Aggragate(CudaPieceFloat a1, CudaPieceFloat a2, CudaPieceFloat a3, CudaPieceFloat b, int batchsize, int m)
        {
            // not implemented
            //BasicMathlib.Matrix_Aggragate(a.MemPtr, b.MemPtr, batchsize, m);
        }

        public void Cosine_Similarity_SubSpace(CudaPieceFloat a, CudaPieceFloat b, CudaPieceFloat c, int labelDim, int BATCHSIZE, int batchsize, int subspaceDim, float eps)
        {
            BasicMathlib.Cosine_Similarity_SubSpace(a.MemPtr, b.MemPtr, c.MemPtr, labelDim, BATCHSIZE, batchsize, subspaceDim, eps);
        }

        public void SoftMax(CudaPieceFloat a, CudaPieceFloat b, int labelDim, int batchsize, float gamma)
        {
            BasicMathlib.SoftMax(a.MemPtr, b.MemPtr, labelDim, batchsize, gamma);
        }

        public void Deriv_Cosine_Subspace(CudaPieceFloat q, CudaPieceFloat d, CudaPieceFloat dcq, CudaPieceFloat dcd, CudaPieceFloat alpha, int act_type, int batchsize, int labelDim, int subspaceDim, float gamma, float eps)
        {
            BasicMathlib.Deriv_Cosine_Subspace(q.MemPtr, d.MemPtr, dcq.MemPtr, dcd.MemPtr, alpha.MemPtr, act_type, batchsize, labelDim, subspaceDim, gamma, eps);
        }

        public void InnerProduct_Similarity(CudaPieceFloat a, CudaPieceFloat b, CudaPieceFloat c, int batchsize, int dimension)
        {
            BasicMathlib.InnerProduct_Similarity(a.MemPtr, b.MemPtr, c.MemPtr, batchsize, dimension);
        }

        public void Deriv_InnerProduct(CudaPieceFloat q, CudaPieceFloat d, CudaPieceFloat dcq, CudaPieceFloat dcd, CudaPieceFloat alpha, int act_type, int batchsize, int Dim, float gamma, float eps)
        {
            BasicMathlib.Deriv_InnerProduct(q.MemPtr, d.MemPtr, dcq.MemPtr, dcd.MemPtr, alpha.MemPtr, act_type, batchsize, Dim, gamma, eps);
        }

        public void Matrix_Add_OFFSET(CudaPieceFloat a, int offset_a, CudaPieceFloat b, int offset_b, int len, float mweight)
        {
            BasicMathlib.Matrix_Add_OFFSET(a.MemPtr, offset_a, b.MemPtr, offset_b, len, mweight);
        }

        public void FillOut_Composite(CudaPieceFloat outputLast, BatchSample_Input data, CudaPieceFloat compData, LookupTab contextLT, CudaPieceFloat context, int d1, int d2, int direction)
        {
            // not implemented
        }

        public void Sparse_Update_Lookup(LookupTab table, LookupTabRunData tableD, int seq1size, int seq2size, int Feature_Dimension, float lr)
        {
            // not implemented
        }

        public void Sparse_Update_Lookup_Update(CudaPieceFloat tabUpdate, LookupTabRunData tableD, int seq1size, int seq2size, int Feature_Dimension, float lr)
        {
            // not implemented
        }
    }

}
