using FourPixCam.Enums;
using MatrixHelper;
using System;

namespace FourPixCam
{
    public class NeuralNet
    {
        public int[] NeuronsPerLayer { get; set; }
        public int L { get; set; }

        public float WeightRange { get; set; }
        public float BiasRange { get; set; }
        public Matrix[] W { get; set; }
        public Matrix[] B { get; set; }
        /// <summary>
        /// input: z, output: a=f(z)
        /// </summary>
        public Func<float,float>[] Activations { get; set; }
        /// <summary>
        /// input: z, t: a'=f'(z)=dadz
        /// </summary>
        public Func<float, float, float>[] Derivations { get; set; }
    }
}
