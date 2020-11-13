﻿using FourPixCam.Enums;
using MatrixHelper;
using System;
using System.Collections.Generic;
using System.Linq;

namespace FourPixCam
{
    internal class DataFactory
    {
        #region ctor & fields

        static Random rnd;
        static Dictionary<Label, Matrix> rawInputs;
        static Dictionary<Label, Matrix> validInputs;
        static Dictionary<Label, Matrix> validOutputs;
        static Sample[] validSamples;

        static DataFactory()
        {
            rnd = RandomProvider.GetThreadRandom();
            rawInputs = GetRawInputs();
            validInputs = GetValidInputs(rawInputs);
            validOutputs = GetValidOutputs();
            validSamples = GetValidSamples();
        }

        #endregion

        internal static Sample[] GetTrainingData(int sampleSize)
        {
            return Enumerable.Range(0, sampleSize)
                .Select(x => GetRandomValidSample())
                .ToArray();
        }
        internal static Sample[] GetTestingData()
        {
            return validSamples;
        }

        #region helper methods

        static Sample[] GetValidSamples()
        {
            var result = new List<Sample>();

            // int vs Label ? .Select(x => (Label)x)?
            var labels = Enum.GetValues(typeof(Label)).ToList<Label>().Skip(1); 
            foreach (var label in labels)
            {
                result.Add(new Sample
                {
                    Label = label,
                    Input = validInputs[label],
                    ExpectedOutput = validOutputs[label]
                });
            }

            return result.ToArray();
        }
        static Dictionary<Label, Matrix> GetRawInputs()
        {
            return new Dictionary<Label, Matrix>
            {
                [Label.AllWhite] = new Matrix(new float[,] {
                    { 0, 0 },
                    { 0, 0 } }),

                [Label.AllBlack] = new Matrix(new float[,] {
                    { 1, 1 },
                    { 1, 1 } }),

                [Label.TopWhite] = new Matrix(new float[,] {
                    { 0, 0 },
                    { 1, 1 } }),

                [Label.TopBlack] = new Matrix(new float[,] {
                    { 1, 1 },
                    { 0, 0 } }),

                [Label.LeftWhite] = new Matrix(new float[,] {
                    { 0, 1 },
                    { 0, 1 } }),

                [Label.LeftBlack] = new Matrix(new float[,] {
                    { 1, 0 },
                    { 1, 0 } }),

                [Label.SlashWhite] = new Matrix(new float[,] {
                    { 1, 0 },
                    { 0, 1 } }),

                [Label.SlashBlack] = new Matrix(new float[,] {
                    { 0, 1 },
                    { 1, 0 } })
            };
        }
        static Dictionary<Label, Matrix> GetValidInputs(Dictionary<Label, Matrix> rawInputs)
        {
            var test = rawInputs.ToDictionary(x => x.Key, x => Operations.ChangeIntoOneColumn(x.Value));
            return test;
        }
        static Dictionary<Label, Matrix> GetValidOutputs()
        {
            return new Dictionary<Label, Matrix>
            {
                [Label.AllWhite] = new Matrix(new float[] { 1, 0, 0, 0 }),

                [Label.AllBlack] = new Matrix(new float[] { 1, 0, 0, 0 }),

                [Label.TopWhite] = new Matrix(new float[] { 0, 1, 0, 0 }),

                [Label.TopBlack] = new Matrix(new float[] { 0, 1, 0, 0 }),

                [Label.LeftWhite] = new Matrix(new float[] { 0, 0, 1, 0 }),

                [Label.LeftBlack] = new Matrix(new float[] { 0, 0, 1, 0 }),

                [Label.SlashWhite] = new Matrix(new float[] { 0, 0, 0, 1 }),

                [Label.SlashBlack] = new Matrix(new float[] { 0, 0, 0, 1 })
            };
        }
        static Sample GetRandomValidSample()
        {
            return validSamples.ElementAt(rnd.Next(0, validInputs.Count() - 1));
        }

        #endregion
    }
}
