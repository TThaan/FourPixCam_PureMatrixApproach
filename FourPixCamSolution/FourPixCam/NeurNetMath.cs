using FourPixCam.Activators;
using FourPixCam.Enums;
using MatrixHelper;
using System;
using System.Linq;
using static MatrixHelper.Operations;

namespace FourPixCam
{
    // Exchange activation type parameters with funcs!?

    // wa: not static but as idisp-instance for each back-prop
    // including stored arrays '..ofLayer[l]'?

    public class NeurNetMath
    {
        public enum CostType
        {
            Undefined, SquaredMeanError
        }

        /// <summary>
        /// Weighted input z=wa+b.
        /// </summary>
        public static Matrix z(Matrix w, Matrix a, Matrix b)
        {
            return ScalarProduct(w, a) + b;
        }
        /// <summary>
        /// Activation function of the weighted input a=f(z).
        /// </summary>
        public static Matrix a(Matrix z, Func<float, float> activation)
        {
            Matrix result = new Matrix(z.m, 1);

            for (int j = 0; j < z.m; j++)
            {
                result[j,1] = activation(z[j, 1]);
            }

            return result;
        }
        /// <summary>
        /// Partial derivation of a with respect to z.
        /// </summary>
        public static Matrix dadz(Matrix a, Matrix z, Func<float, float, float> derivationOfActivation)
        {
            return Partial(a, z, derivationOfActivation);
        }
        public static Matrix C(Matrix a, Matrix t, Func<float, float, float> c0)
        {
            Matrix result = new Matrix(a.m, 1);

            for (int j = 0; j < a.m; j++)
            {
                result[j, 1] = c0(a[j, 1], t[j, 1]);
            }

            return result;
        }
        public static float CTotal(Matrix a, Matrix t, Func<float, float, float> c0)
        {
            // CTotal = total or averaged (i.e. sum divided by a.m)?
            return C(a, t, c0).Sum();
        }
        /// <summary>
        /// = delta * w
        /// </summary>
        /// <returns></returns>
        public static Matrix dCda(float C, Matrix a, Func<float, float, float> derivationOfActivation)
        {
            Matrix result = a.Transpose;

            for (int j = 0; j < a.m; j++)
            {
                result[1, j] = derivationOfActivation(C, a[1, j]);
            }

            return result;
        }
        /// <summary>
        /// Delta-matrix of the output layer: 
        /// delta=dCd(z_output)= 
        /// </summary>
        public static Matrix delta_Output(float C, Matrix z_Output, Func<float, float, float> derivationOfActivation)
        {
            return Partial(C, z_Output, derivationOfActivation);
        }
        /// <summary>
        /// Delta-matrix of a hidden layer: 
        /// delta=dCd(z_hidden)= 
        /// </summary>
        public static Matrix delta_Hidden(Matrix delta, Matrix w, Matrix a, Matrix z, Func<float, float, float> derivationOfActivation)
        {
            Matrix dCda = delta * w;            
            return dCda * dadz(a, z, derivationOfActivation);
        }
        /// <summary>
        /// delta^l * a^(l-1) = dC/dw^l
        /// </summary>
        public static Matrix GetCorrectedWeights(Matrix w, Matrix a, Matrix delta, float learningRate)
        {
            // = dCda*dadz*dzdw = error^L * a^(L-1) ??


            Matrix dCdw = delta.Transpose * a.Transpose;
            return w - learningRate * dCdw;

            //Matrix result = new Matrix(w.m, w.n);
            //for (int j = 0; j < w.m; j++)
            //{
            //    for (int k = 0; k < w.n; k++)
            //    {
            //        result[j, k] = w[j, k] - a[k, 0] * delta[j, 0];
            //    }
            //}

            //return result;
        }
        public static Matrix GetCorrectedBiases(Matrix b, Matrix delta, float learningRate)
        {
            return b - learningRate * delta.Transpose;
        }
    }
}
