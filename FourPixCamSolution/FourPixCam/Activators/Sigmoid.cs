﻿using MatrixHelper;
using System;
using System.Linq;

namespace FourPixCam.Activators
{
    public class Sigmoid
    {
        /// <summary>
        /// Activation ('squashing') function of any weighted input neuron z.
        /// </summary>
        public static float a(float z)
        {
            return 1 / (1 + (float)Math.Exp(-z));
        }
        /// <summary>
        /// Activation ('squashing') function of the weighted input matrix z.
        /// </summary>
        public static Matrix a(Matrix z)
        {
            return new Matrix(
                z.Select(x => 1 / (1 + (float)Math.Exp(-x)))
                .ToArray());
        }
        /// <summary>
        /// Derivation of the activation ('squashing') function with respect to any weighted input z.
        /// </summary>
        public static float dadz(float z)
        {
            return z * (1 - z);
        }
        /// <summary>
        /// Partial derivation of the activation ('squashing') function with respect to the weighted input z.
        /// </summary>
        public static Matrix dadz(Matrix z)
        {
            return new Matrix(z.Select(x => x * (1 - x)).ToArray())
                .Transpose;
        }
    }
}
