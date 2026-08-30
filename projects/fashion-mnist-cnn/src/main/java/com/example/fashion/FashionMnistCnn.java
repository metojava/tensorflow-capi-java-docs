package com.example.fashion;

import com.example.tfcapi.*;

import java.nio.file.Path;

/**
 * Project 6 - Fashion-MNIST clothing classification with a CNN that showcases
 * DepthwiseConv2D (separable convolutions).
 *
 * A plain Conv2D(32,3,3) applied to 32 channels learns 32*32 3x3 kernels
 * (9,216 weights); a depthwise conv learns ONE 3x3 filter per channel
 * (288 weights) - ~32x fewer parameters - yet still adds useful spatial
 * feature mixing before the dense head.
 *
 * Usage: mvn exec:java "-Dexec.args=DATADIR EPOCHS BATCH"
 *        or named flags: "--datadir DIR --epochs N --batch B".
 */
public final class FashionMnistCnn {

    private static final int CLASSES = 10;
    private static final String[] NAMES = Metrics.FASHION_CLASS_NAMES;

    public static void main(String[] args) throws Exception {
        // ---------- 1. Parse arguments ------------------------------------
        String dataDir = "C:\\Users\\dell\\Desktop\\data\\fashion";
        int epochs = 8;
        int batch = 128;
        if (args.length > 0 && !args[0].startsWith("--")) {
            // positional form: DATADIR [EPOCHS] [BATCH]
            dataDir = args[0];
            if (args.length > 1) epochs = Integer.parseInt(args[1]);
            if (args.length > 2) batch = Integer.parseInt(args[2]);
        } else {
            for (int i = 0; i < args.length; i += 2) {
                if ("--datadir".equals(args[i]) && i + 1 < args.length) dataDir = args[i + 1];
                else if ("--epochs".equals(args[i]) && i + 1 < args.length) epochs = Integer.parseInt(args[i + 1]);
                else if ("--batch".equals(args[i]) && i + 1 < args.length) batch = Integer.parseInt(args[i + 1]);
                else throw new IllegalArgumentException(
                        "usage: [datadir epochs batch] or [--datadir D --epochs N --batch B]");
            }
        }
        System.out.printf("dataDir=%s epochs=%d batch=%d%n", dataDir, epochs, batch);

        // ---------- 2. Load the IDX .gz files -----------------------------
        IdxDataset.Images trI = IdxDataset.loadImages(
                Path.of(dataDir, "train-images-idx3-ubyte.gz").toString());
        IdxDataset.Labels trL = IdxDataset.loadLabels(
                Path.of(dataDir, "train-labels-idx1-ubyte.gz").toString());
        IdxDataset.Images teI = IdxDataset.loadImages(
                Path.of(dataDir, "t10k-images-idx3-ubyte.gz").toString());
        IdxDataset.Labels teL = IdxDataset.loadLabels(
                Path.of(dataDir, "t10k-labels-idx1-ubyte.gz").toString());

        float[] trainX = IdxDataset.normalize(trI.pixels());   // uint8 -> float in [0,1]
        int[] trainY = toInts(trL.labels());
        float[] testX = IdxDataset.normalize(teI.pixels());
        int[] testY = toInts(teL.labels());
        System.out.printf("Loaded %d train / %d test 28x28 grayscale images%n",
                trI.count(), teI.count());

        // ---------- 3. Build the separable-convolution CNN -----------------
        try (Sequential net = new Sequential(Optimizer.adam(0.001f),
                new Input4D(28, 28, 1),
                new Conv2D(32, 3, 3, 1, 1, Padding.SAME),    // learn wide features
                new ReLU(),
                new DepthwiseConv2D(32, 3, 3, 1, 1, Padding.SAME), // cheap per-channel refine
                new ReLU(),
                new MaxPool2D(2, 2, 2, 2, Padding.VALID),    // 28 -> 14
                new Flatten(),
                new Dense(128), new ReLU(),
                new Dropout(0.25f),
                new Dense(CLASSES))) {                       // LAST layer = Dense(numClasses)

            net.fit(trainX, trainY, epochs, batch, testX, testY, true);

            // ---------- 4. Evaluate on the 10k held-out images ------------
            float[] probs = net.predict(testX);
            double loss = 0;
            int correct = 0;
            int[][] conf = new int[CLASSES][CLASSES];
            for (int i = 0; i < testY.length; i++) {
                int pred = Metrics.argmax(probs, i * CLASSES, CLASSES);
                conf[testY[i]][pred]++;
                if (pred == testY[i]) correct++;
                loss += -Math.log(Math.max(probs[i * CLASSES + testY[i]], 1e-12));
            }
            Metrics.printReport(loss / testY.length,
                    (double) correct / testY.length, conf, NAMES);
            System.out.println("Note which classes confuse each other - shirts/coats/pullovers "
                    + "share a silhouette and are much harder than digits.");

            // ---------- 5. Show a few textual sample predictions ----------
            for (int i = 0; i < 10; i++) {
                int pred = Metrics.argmax(probs, i * CLASSES, CLASSES);
                System.out.printf("sample %d: true=%s predicted=%s %s%n",
                        i, NAMES[testY[i]], NAMES[pred], pred == testY[i] ? "" : "(MISS)");
            }

            net.saveTo(Path.of("target", "fashion-model.bin"));
            System.out.println("Model saved to target/fashion-model.bin");
        }
    }

    static int[] toInts(byte[] labels) {
        int[] out = new int[labels.length];
        for (int i = 0; i < labels.length; i++) out[i] = labels[i] & 0xFF;
        return out;
    }

    private FashionMnistCnn() {}
}
