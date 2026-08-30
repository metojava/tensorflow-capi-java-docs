package com.example.iris;

import com.example.tfcapi.*;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Project 1 - Iris flower classification.
 *
 * A classic beginner project: predict the iris species (setosa, versicolor,
 * virginica) from four measurements of the flower. Demonstrates loading a CSV,
 * feature standardization, a train/test split, an MLP trained with Adam and
 * a confusion-matrix evaluation.
 */
public final class IrisClassifier {

    private static final String[] SPECIES = {"setosa", "versicolor", "virginica"};

    public static void main(String[] args) throws Exception {
        // ---------- 1. Load and parse the dataset -------------------------
        List<String> lines = readLines("/data/iris.data");
        int n = lines.size();
        float[] x = new float[n * 4];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            String[] p = lines.get(i).split(",");
            for (int f = 0; f < 4; f++) x[i * 4 + f] = Float.parseFloat(p[f]);
            y[i] = switch (p[4].trim()) {
                case "Iris-setosa" -> 0;
                case "Iris-versicolor" -> 1;
                default -> 2;
            };
        }
        System.out.println("Loaded " + n + " iris flowers");

        // ---------- 2. Seeded shuffle + train/test split (105/45) --------
        int[] order = seededOrder(n, 42);
        float[] trainX = new float[105 * 4]; int[] trainY = new int[105];
        float[] testX  = new float[(n - 105) * 4]; int[] testY = new int[n - 105];
        for (int i = 0; i < n; i++) {
            if (i < 105) { copyRow(x, y, order[i], trainX, trainY, i); }
            else         { copyRow(x, y, order[i], testX, testY, i - 105); }
        }

        // ---------- 3. Standardize features with TRAIN statistics --------
        float[] mean = columnMean(trainX, 105);
        float[] std = columnStd(trainX, 105, mean);
        standardize(trainX, mean, std);
        standardize(testX, mean, std);

        // ---------- 4. Build and train the MLP ---------------------------
        try (Sequential net = new Sequential(Optimizer.adam(0.01f),
                new Input(4),
                new Dense(16), new ReLU(),
                new Dense(3))) {

            net.fit(trainX, trainY, 150, 16, testX, testY, true);

            // ---------- 5. Evaluate on the held-out test set -------------
            float[] probs = net.predict(testX);
            double loss = 0; int correct = 0;
            int[][] conf = new int[3][3];
            for (int i = 0; i < testY.length; i++) {
                int pred = Metrics.argmax(probs, i * 3, 3);
                conf[testY[i]][pred]++;
                if (pred == testY[i]) correct++;
                loss += -Math.log(Math.max(probs[i * 3 + testY[i]], 1e-12));
            }
            Metrics.printReport(loss / testY.length, (double) correct / testY.length,
                    conf, SPECIES);

            net.saveTo(Path.of("target", "iris-model.bin"));
            System.out.println("Model saved to target/iris-model.bin");
        }
    }

    /* ------------------------- helper methods --------------------------- */

    /** Reads a classpath resource into non-blank trimmed lines. */
    static List<String> readLines(String resource) throws Exception {
        try (InputStream in = IrisClassifier.class.getResourceAsStream(resource)) {
            if (in == null) throw new IllegalStateException("resource not found: " + resource);
            List<String> out = new ArrayList<>();
            BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) out.add(line);
            }
            return out;
        }
    }

    static int[] seededOrder(int n, long seed) {
        int[] o = new int[n];
        for (int i = 0; i < n; i++) o[i] = i;
        Random r = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1), t = o[i]; o[i] = o[j]; o[j] = t;
        }
        return o;
    }

    static void copyRow(float[] src, int[] srcY, int row,
                        float[] dst, int[] dstY, int dstRow) {
        System.arraycopy(src, row * 4, dst, dstRow * 4, 4);
        dstY[dstRow] = srcY[row];
    }

    static float[] columnMean(float[] data, int rows) {
        float[] m = new float[4];
        for (int i = 0; i < rows; i++)
            for (int f = 0; f < 4; f++) m[f] += data[i * 4 + f];
        for (int f = 0; f < 4; f++) m[f] /= rows;
        return m;
    }

    static float[] columnStd(float[] data, int rows, float[] mean) {
        float[] s = new float[4];
        for (int i = 0; i < rows; i++)
            for (int f = 0; f < 4; f++) {
                float d = data[i * 4 + f] - mean[f];
                s[f] += d * d;
            }
        for (int f = 0; f < 4; f++) s[f] = (float) Math.sqrt(s[f] / Math.max(rows - 1, 1));
        return s;
    }

    static void standardize(float[] data, float[] mean, float[] std) {
        for (int i = 0; i < data.length / 4; i++)
            for (int f = 0; f < 4; f++)
                data[i * 4 + f] = (data[i * 4 + f] - mean[f]) / (std[f] + 1e-8f);
    }

    private IrisClassifier() {}
}
