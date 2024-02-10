from gan.fcgan import ConvGenerator, ConvDiscriminator, FCGanConfig, FCGan


def test_ConvGenerator():
    import torch

    inp_channels = 3
    out_channels = 5
    latent_channels = 32
    img_size = 256
    batch_size = 7
    x = torch.randn(batch_size, inp_channels, img_size, img_size)
    model = ConvGenerator(inp_channels, out_channels, latent_channels)
    print(model)
    y = model(x)
    assert y.shape == (batch_size, out_channels, img_size, img_size)


def test_ConvDiscriminator():
    import torch

    inp_channels = 5
    img_size = 256
    batch_size = 7
    x = torch.randn(batch_size, inp_channels, img_size, img_size)
    disc = ConvDiscriminator(inp_channels, [64, 128, 256, 512])
    print(disc)
    y = disc(x)
    print(y.shape)


def test_DCGan():
    import torch

    config = FCGanConfig(
        "",
        "dcgan",
        inp_features=3,
        out_features=5,
        latent_features=64,
        downsample=2,
        residuals=9,
        n=1,
        p=0,
        blocks=(64, 128, 256, 512),
    )
    model = FCGan(config)
    print(model)
    x = torch.randn(7, 3, 256, 256)
    yg = model.G(x)
    yd = model.D(yg)
    print(yg.shape)
    print(yd.shape)


if __name__ == "__main__":
    test_ConvGenerator()
    test_ConvDiscriminator()
    test_DCGan()
