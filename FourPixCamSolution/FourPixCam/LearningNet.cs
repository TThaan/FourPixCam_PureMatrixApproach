﻿using FourPixCam.Activators;
using FourPixCam.CostFunctions;
using MatrixHelper;
using System;
using System.Linq;
using static FourPixCam.NeurNetMath;

namespace FourPixCam
{
    public class LearningNet
    {
        #region fields

        // readonly Random rnd = RandomProvider.GetThreadRandom();        
        NeuralNet net;

        #endregion

        #region ctor

        public LearningNet(NeuralNet net)      // param jasonFile
        {
            this.net = net;

            CostType = CostType.SquaredMeanError;

            Z = new Matrix[net.L];
            A = new Matrix[net.L];
            dadz_OfLayer = new Matrix[net.L];
            Delta = new Matrix[net.L];
        }

        #region helper methods

        #endregion

        #endregion

        #region properties

        // public Matrix x { get; set; }

        /// <summary>
        /// expected output
        /// </summary>
        public Matrix t { get; set; }   // redundant?
        public float C { get; set; }
        /// <summary>
        /// total value (= wa + b)
        /// </summary>
        public Matrix[] Z { get; set; }
        /// <summary>
        /// activation (= f(z))
        /// </summary>
        public Matrix[] A { get; set; }
        public Matrix[] dadz_OfLayer { get; set; }  // => Matrix.Partial(f, a);
        public Matrix[] Delta { get; set; }
        public Matrix[] F { get; set; } // => activations[]

        public CostType CostType { get; set; }
        public float LastCost { get; set; } // redundant?


        #endregion

        #region methods

        public Matrix FeedForwardAndGetOutput(Matrix input)
        {
            Console.WriteLine("\n    *   *   *   *  *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   \n");
            Console.WriteLine($"                                        F E E D   F O R W A R D");
            Console.WriteLine("\n    *   *   *   *  *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   \n");
            Console.WriteLine();

            // wa: Separate inp layer from 'layers' ?!
            A[0] = input.DumpToConsole($"\nA[0] = "); //new Matrix(input.ToArray());
            Console.WriteLine("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");


            // iterate over layers (skip input layer)
            for (int i = 1; i < net.L; i++)
            {
                Z[i] = NeurNetMath.z(net.W[i].DumpToConsole($"\nW{i} = "), A[i - 1].DumpToConsole($"\nA{i-1} = "), net.B[i].DumpToConsole($"\nB{i} = ")).DumpToConsole($"\nZ{i} = ");
                A[i] = NeurNetMath.a(Z[i], net.Activations[i]).DumpToConsole($"\nA{i} = ");
                Console.WriteLine("\n    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   ");
            }

            return A.Last();
        }
        public void BackPropagate(Matrix y, float learningRate)
        {
            Console.WriteLine("\n    *   *   *   *  *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   \n");
            Console.WriteLine($"                                        B A C K P R O P A P A G A T I O N");
            Console.WriteLine("\n    *   *   *   *  *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   \n");
            Console.WriteLine();

            // debug
            var c = NeurNetMath.C(A[net.L - 1], y, SquaredMeanError.CostFunction)
                .DumpToConsole($"\n{CostType.SquaredMeanError} C =");
            var cTotal = NeurNetMath.CTotal(A[net.L - 1], y, SquaredMeanError.CostFunction);
            Console.WriteLine($"\nCTotal = {cTotal}\n");

            // Iterate backwards over each layer (skip input layer).
            for (int l = net.L - 1; l > 0; l--)
            {
                dadz_OfLayer[l] = NeurNetMath.dadz(A[l], Z[l], net.Derivations[l])
                    .DumpToConsole($"\ndadz{l} =");
                Matrix error;

                if (l == net.L - 1)
                {
                    // .. and C0 instead of a[i] and t as parameters here?
                    error = NeurNetMath.delta_Output(cTotal, Z[l], SquaredMeanError.DerivationOfCostFunction);
                }
                else
                {
                    error = NeurNetMath.delta_Hidden(Delta[l + 1], net.W[l + 1], A[l], Z[l], net.Derivations[l]);
                }

                Delta[l] = error.DumpToConsole($"\ndelta{l} =");
            }

            // Adjust weights and biases (skip input layer).
            for (int l = 1; l < net.L; l++)
            {
                // E or delta ?
                net.W[l] = GetCorrectedWeights(net.W[l], A[l-1], Delta[l], learningRate)
                        .DumpToConsole($"\nadjusted w{l} =");
                net.B[l] = GetCorrectedWeights(net.B[l], A[l - 1], Delta[l], learningRate)
                        .DumpToConsole($"\nadjusted b{l} =");
            }
        }

        #region helper methods

        #endregion

        #endregion

        #region helper methods

        #endregion
    }
}
