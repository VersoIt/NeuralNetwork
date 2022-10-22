using System;

namespace NeuralNetwork.Tools
{
    public static class MathFunction
    {

        public static double GetResultOfSigmoid(this double x) => 1.0 / (1.0 + Math.Pow(Math.E, -x));

        public static double GetResultOfSigmoidDx(double x)
        {
            var sigmoid = GetResultOfSigmoid(x);
            return sigmoid / (1 - sigmoid);
        }
    }
}
