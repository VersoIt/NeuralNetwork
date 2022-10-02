using System.Drawing;

namespace PictureConverter
{
    public class ImageBinarizer : Abstract.IConvertible<Bitmap>
    {
        public const int Boundary = 128;

        private Bitmap _image;

        public ImageBinarizer(Bitmap image)
        {
            _image = image;

            Width = image.Width;
            Height = image.Height;
        }

        public int Width { get; init; }
        public int Height { get; init; }

        public Bitmap Convert()
        {
            Bitmap result = new Bitmap(Width, Height);
            for (int x = 0; x < _image.Width; ++x)
            {
                for (int y = 0; y < _image.Height; ++y)
                {
                    result.SetPixel(x, y, GetBrightnessPixel(_image.GetPixel(x, y)));
                }
            }
            return result;
        }

        private Color GetBrightnessPixel(Color pixel)
        {
            double transition = 0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B;
            return transition < Boundary ? Color.White : Color.Black;
        }
    }
}