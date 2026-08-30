package com.example.mnist;

import com.example.tfcapi.*;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Project 4 - MNIST handwritten digit classification with a CNN.
 *
 * Loads the classic MNIST IDX files (gzip compressed), normalizes pixels to
 * [0,1] and trains a small convolutional network:
 *
 *   Input4D(28,28,1) -> Conv2D(16) -> ReLU -> MaxPool(2)
 *                    -> Conv2D(32) -> ReLU -> MaxPool(2)
 *                    -> Flatten -> Dense(128) -> ReLU -> Dropout -> Dense(10)
 *
 * Spatial math with SAME padding and stride-1 convs, stride-2 pooling:
 *   28x28 -> conv SAME -> 28x28x16 -> pool -> 14x14x16
 *         -> conv SAME -> 14x14x32 -> pool ->  7x7x32 -> flatten 1568
 *
 * Command line (all optional):
 *   args[0] = data dir holding the four .gz IDX files
 *             (default C:\Users\dell\Desktop\data\mnist)
 *   args[1] = epochs   (default 3)
 *   args[2] = batch    (default 128)
 * Flags --datadir/--epochs/--batch are also accepted.
 */
public final class MnistDigitCnn {

    private static final String[] DIGITS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

    public static void main(String[] args) throws Exception {
        // ---------- 1. Parse arguments (positional or --flags) ------------
        String dataDir = "C:\\Users\\dell\\Desktop\\data\\mnist";
        int epochs = 3;
        int batchSize = 128;
        List<String> pos = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--datadir", "--data" -> dataDir = args[++i];
                case "--epochs" -> epochs = Integer.parseInt(args[++i]);
                case "--batch" -> batchSize = Integer.parseInt(args[++i]);
                default -> pos.add(args[i]);
            }
        }
        if (pos.size() > 0) dataDir = pos.get(0);
        if (pos.size() > 1) epochs = Integer.parseInt(pos.get(1));
        if (pos.size() > 2) batchSize = Integer.parseInt(pos.get(2));
        System.out.printf("dataDir=%s epochs=%d batch=%d%n", dataDir, epochs, batchSize);

        // ---------- 2. Load IDX files and normalize to [0,1] --------------
        IdxDataset.Images trIm = IdxDataset.loadImages(dataDir + "\\train-images-idx3-ubyte.gz");
        IdxDataset.Labels trLb = IdxDataset.loadLabels(dataDir + "\\train-labels-idx1-ubyte.gz");
        IdxDataset.Images teIm = IdxDataset.loadImages(dataDir + "\\t10k-images-idx3-ubyte.gz");
        IdxDataset.Labels teLb = IdxDataset.loadLabels(dataDir + "\\t10k-labels-idx1-ubyte.gz");
        System.out.println("train images=" + trIm.count() + " test images=" + teIm.count());

        // normalize(byte[]) maps unsigned bytes 0..255 to floats 0..1
        float[] trainX = IdxDataset.normalize(trIm.pixels());
        float[] testX = IdxDataset.normalize(teIm.pixels());
        int[] trainY = toInts(trLb.labels());
        int[] testY = toInts(teLb.labels());

        // ---------- 3. Build the CNN --------------------------------------
        try (Sequential net = new Sequential(Optimizer.adam(0.001f),
                new Input4D(28, 28, 1),
                new Conv2D(16, 3, 3, 1, 1, Padding.SAME), new ReLU(),
                new MaxPool2D(2, 2, 2, 2, Padding.VALID),
                new Conv2D(32, 3, 3, 1, 1, Padding.SAME), new ReLU(),
                new MaxPool2D(2, 2, 2, 2, Padding.VALID),
                new Flatten(),
                new Dense(128), new ReLU(),
                new Dropout(0.25f),
                new Dense(10))) {

            System.out.println("params=" + net.getParams().length + " inputElems=" + net.inputElems());

            // ---------- 4. Train (validation on the 10k test set) ---------
            net.fit(trainX, trainY, epochs, batchSize, testX, testY, true);

            // ---------- 5. Evaluate: accuracy + 10x10 confusion matrix ----
            float[] probs = net.predict(testX);
            double loss = 0;
            int correct = 0;
            int[][] conf = new int[10][10];
            for (int i = 0; i < testY.length; i++) {
                int pred = Metrics.argmax(probs, i * 10, 10);
                conf[testY[i]][pred]++;
                if (pred == testY[i]) correct++;
                loss += -Math.log(Math.max(probs[i * 10 + testY[i]], 1e-12));
            }
            double acc = (double) correct / testY.length;
            Metrics.printReport(loss / testY.length, acc, conf, DIGITS);

            // ---------- 6. Ten sample predictions printed textually -------
            System.out.println("\nSample predictions (first 10 test images):");
            for (int i = 0; i < 10; i++) {
                int pred = Metrics.argmax(probs, i * 10, 10);
                float p = probs[i * 10 + pred];
                System.out.printf("image %2d: true=%d predicted=%d confidence=%.3f %s%n",
                        i, testY[i], pred, p, pred == testY[i] ? "OK" : "MISS");
            }

            net.saveTo(Path.of("target", "mnist-cnn-model.bin"));
            System.out.println("Model saved to target/mnist-cnn-model.bin");
        }
    }

    /** IDX labels are unsigned bytes; widen them to ints for fit()/fit labels. */
    static int[] toInts(byte[] raw) {
        int[] y = new int[raw.length];
        for (int i = 0; i < raw.length; i++) y[i] = raw[i] & 0xFF;
        return y;
    }

    private MnistDigitCnn() {}
}
