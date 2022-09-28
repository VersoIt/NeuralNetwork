namespace Neuron.Tools
{
    public static class Matrix
    {
        public static T[] GetRow<T>(T[,] array, int row) => Enumerable.Range(0, array.GetLength(1)).Select(column => array[row, column]).ToArray();

        public static T[] GetColumn<T>(T[,] array, int column) => Enumerable.Range(0, array.GetLength(0)).Select(row => array[row, column]).ToArray();
    }
}
