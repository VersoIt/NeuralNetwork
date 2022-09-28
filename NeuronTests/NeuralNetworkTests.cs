using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuron;
using Neuron.Tools;
using System.Globalization;
using System.Net.Http.Headers;

namespace NeuronTests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1 };
            var inputs = new double[,]
            {
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 },
                { 0, 0, 1, 0 },
                { 0, 0, 1, 1 },
                { 0, 1, 0, 0 },
                { 0, 1, 0, 1 },
                { 0, 1, 1, 0 },
                { 0, 1, 1, 1 },
                { 1, 0, 0, 0 },
                { 1, 0, 0, 1 },
                { 1, 0, 1, 0 },
                { 1, 0, 1, 1 },
                { 1, 1, 0, 0 },
                { 1, 1, 0, 1 },
                { 1, 1, 1, 0 },
                { 1, 1, 1, 1 }
            };

            var topology = new Topology(4, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetwork(topology);
            var difference = neuralNetwork.Learn(outputs, inputs, 100000);

            var results = new List<double>();
            for (int i = 0; i < outputs.Length; ++i)
            {
                var row = Matrix.GetRow(inputs, i);
                var res = neuralNetwork.FeedForward(row).Output;
                results.Add(res);
            }

            for (int i = 0; i < results.Count; ++i)
            {
                var expected = Math.Round(outputs[i], 3);
                var actual = Math.Round(results[i], 3);
                Assert.AreEqual(expected, actual);
            }
        }

        [TestMethod()]
        public void DatasetTest()
        {
            var outputs = new List<double>();
            var inputs = new List<double[]>();

            using (var stream = new StreamReader("heart.csv"))
            {
                var header = stream.ReadLine();
                while (!stream.EndOfStream)
                {
                    var row = stream.ReadLine();
                    if (row != null)
                    {
                        var values = row.Split(',').Select(s => Double.Parse(s.Replace(".", CultureInfo.CurrentCulture.NumberFormat.NumberDecimalSeparator))).ToArray();
                        var output = values.Last();
                        var input = values.Take(values.Length - 1).ToArray();

                        outputs.Add(output);
                        if (input != null)
                            inputs.Add(input);
                    }
                }

                var inputSignals = new double[inputs.Count, inputs[0].Length];
                for (int i = 0; i < inputSignals.GetLength(0); ++i)
                {
                    for (int j = 0; j < inputSignals.GetLength(1); ++j)
                    {
                        inputSignals[i, j] = inputs[i][j];
                    }
                }

                var topology = new Topology(inputSignals.GetLength(1), 1, 0.1, outputs.Count() / 2);
                var neuralNetwork = new NeuralNetwork(topology);
                var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 5000);

                var results = new List<double>();
                for (int i = 0; i < outputs.Count(); ++i)
                {
                    var res = neuralNetwork.FeedForward(inputs[i]).Output;
                    results.Add(res);
                }

                for (int i = 0; i < results.Count; ++i)
                {
                    var expected = Math.Round(outputs[i], 3);
                    var actual = Math.Round(results[i], 3);
                    Assert.AreEqual(expected, actual);
                }
            }
        }
    }
}