package com.example.wine;

import com.example.tfcapi.*;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Project 5 - Red wine quality classification from physicochemical tests.
 *
 * The UCI wine-quality dataset (1599 red wines) ships with the project at
 * src/main/resources/data/winequality-red.csv (semicolon separated). Each
 * row holds 11 numeric features plus an integer quality score 3..8.
 *
 * Because raw quality counts are heavily imbalanced, we remap them into
 * three bands: low (&le;4), medium (5-6), high (&ge;7). Most wines fall into
 * "medium", so per-class recall in the report matters more than raw
 * accuracy - a model predicting "medium" for everything would score ~82%
 * while being useless.
 */
public final class WineQualityClassifier {

    private static final int FEATURES = 11;
    private static final int CLASSES = 3;

    private static final String[] BAND_NAMES = {"low", "medium", "high"};

    public static void main(String[] args) throws Exception {
        // ---------- 1. Load and parse the CSV ----------------------------
        List<String> lines = readLines("/data/winequality-red.csv");
        // First line is the header ("fixed acidity";...;"quality") - skip it.
        int n = lines.size() - 1;
        float[] x = new float[n * FEATURES];
        int[] y = new int[n];
        int[] qualityCounts = new int[CLASSES];
        for (int i = 0; i < n; i++) {
            String[] p = lines.get(i + 1).split(";");
            for (int f = 0; f < FEATURES; f++) {
                x[i * FEATURES + f] = Float.parseFloat(p[f].trim());
            }
            int quality = Integer.parseInt(p[FEATURES].trim());
            y[i] = remapQuality(quality);
            qualityCounts[y[i]]++;
        }
        System.out.println("Loaded " + n + " wines. Class distribution:"
                + " low=" + qualityCounts[0]
                + " medium=" + qualityCounts[1]
                + " high=" + qualityCounts[2]);

        // ---------- 2. Seeded shuffle + 70/30 split ----------------------
        int[] order = seededOrder(n, 42);
        int trainN = (int) Math.round(n * 0.7);          // 1120
        int testN = n - trainN;                          // 479
        float[] trainX = new float[trainN * FEATURES]; int[] trainY = new int[trainN];
        float[] testX = new float[testN * FEATURES];   int[] testY = new int[testN];
        for (int i = 0; i < n; i++) {
            if (i < trainN) copyRow(x, y, order[i], trainX, trainY, i, FEATURES);
            else copyRow(x, y, order[i], testX, testY, i - trainN, FEATURES);
        }

        // ---------- 3. Standardize with TRAIN statistics -----------------
        float[] mean = columnMean(trainX, trainN);
        float[] std = columnStd(trainX, trainN, mean);
        standardize(trainX, mean, std);
        standardize(testX, mean, std);

        // ---------- 4. Build and train the network -----------------------
        try (Sequential net = new Sequential(Optimizer.adam(0.005f),
                new Input(FEATURES),
                new Dense(32), new ReLU(), new BatchNorm1D(), new Dropout(0.2f),
                new Dense(16), new ReLU(),
                new Dense(CLASSES))) {

            net.fit(trainX, trainY, 120, 32, testX, testY, true);

            // ---------- 5. Evaluate: accuracy + confusion report ---------
            float[] probs = net.predict(testX);
            double loss = 0;
            int correct = 0;
            int[][] conf = new int[CLASSES][CLASSES];
            for (int i = 0; i < testN; i++) {
                int pred = Metrics.argmax(probs, i * CLASSES, CLASSES);
                conf[testY[i]][pred]++;
                if (pred == testY[i]) correct++;
                loss += -Math.log(Math.max(probs[i * CLASSES + testY[i]], 1e-12));
            }
            Metrics.printReport(loss / testN, (double) correct / testN,
                    conf, BAND_NAMES);
            System.out.println("(Imbalanced classes: check the per-class recall "
                    + "above, not just overall accuracy.)");
        }
    }

    /* ------------------------- helper methods --------------------------- */

    /**
     * Remaps the 10-point quality scale onto three balanced-enough bands:
     * quality &le; 4 -> low(0), 5-6 -> medium(1), &ge; 7 -> high(2).
     */
    static int remapQuality(int quality) {
        if (quality <= 4) return 0;
        if (quality <= 6) return 1;
        return 2;
    }

    /** Reads a classpath resource into non-blank trimmed lines. */
    static List<String> readLines(String resource) throws Exception {
        try (InputStream in =
                     WineQualityClassifier.class.getResourceAsStream(resource)) {
            if (in == null) throw new IllegalStateException(
                    "resource not found: " + resource);
            List<String> out = new ArrayList<>();
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(in, StandardCharsets.UTF_8));
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
                        float[] dst, int[] dstY, int dstRow, int features) {
        System.arraycopy(src, row * features, dst, dstRow * features, features);
        dstY[dstRow] = srcY[row];
    }

    static float[] columnMean(float[] data, int rows) {
        float[] m = new float[FEATURES];
        for (int i = 0; i < rows; i++)
            for (int f = 0; f < FEATURES; f++) m[f] += data[i * FEATURES + f];
        for (int f = 0; f < FEATURES; f++) m[f] /= rows;
        return m;
    }

    static float[] columnStd(float[] data, int rows, float[] mean) {
        float[] s = new float[FEATURES];
        for (int i = 0; i < rows; i++)
            for (int f = 0; f < FEATURES; f++) {
                float d = data[i * FEATURES + f] - mean[f];
                s[f] += d * d;
            }
        for (int f = 0; f < FEATURES; f++)
            s[f] = (float) Math.sqrt(s[f] / Math.max(rows - 1, 1));
        return s;
    }

    static void standardize(float[] data, float[] mean, float[] std) {
        for (int i = 0; i < data.length / FEATURES; i++)
            for (int f = 0; f < FEATURES; f++)
                data[i * FEATURES + f] =
                        (data[i * FEATURES + f] - mean[f]) / (std[f] + 1e-8f);
    }

    private WineQualityClassifier() {}
}
