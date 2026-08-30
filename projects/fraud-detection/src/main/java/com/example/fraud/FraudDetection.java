package com.example.fraud;

import com.example.tfcapi.*;

import java.nio.file.Path;
import java.util.Random;

/**
 * Project 9 - Credit-card fraud detection on synthetic transactions.
 *
 * Demonstrates the classic class-imbalance trap: a 97%-majority "normal"
 * baseline looks impressive while catching zero frauds, so we evaluate with
 * an explicit fraud precision/recall breakdown and then shift the decision
 * threshold to trade precision for recall.
 */
public final class FraudDetection {

    /** Feature layout of one transaction row. */
    // 0: amount (USD)          1: hour of day (0..23)   2: distanceFromHome
    // 3: online (0/1)          4: ratioToMedianPrice
    private static final int FEATURES = 5;
    private static final String[] NAMES = {"normal", "fraud"};
    private static final int FRAUD = 1;

    public static void main(String[] args) throws Exception {
        // ---------- 1. Generate the synthetic dataset (seed 7) ------------
        Random r = new Random(7);
        int nNormal = 6000, nFraud = 180;
        int n = nNormal + nFraud;
        float[] x = new float[n * FEATURES];
        int[] y = new int[n];

        for (int i = 0; i < nNormal; i++) {
            writeRow(x, y, i,
                    lognormalAmount(r, 1.0),          // typical purchase
                    daytimeHour(r),                   // shoppers are awake
                    Math.abs(r.nextGaussian()),       // near home
                    r.nextDouble() < 0.15 ? 1 : 0,    // 15% online purchases
                    (float) (1 + r.nextGaussian() * 0.2),
                    0);
        }
        for (int j = 0; j < nFraud; j++) {
            int i = nNormal + j;
            writeRow(x, y, i,
                    lognormalAmount(r, 5 + 15 * r.nextDouble()), // big-ticket
                    oddHour(r),                       // 22..05 skew
                    Math.abs(r.nextGaussian()) * 3,   // far from home
                    r.nextDouble() < 0.70 ? 1 : 0,    // card-not-present
                    (float) (1 + r.nextGaussian() * 0.2),
                    1);
        }
        System.out.printf("Generated %d transactions (%d fraud = %.2f%%)%n",
                n, nFraud, 100.0 * nFraud / n);

        // ---------- 2. Seeded shuffle + 70/30 split -----------------------
        int[] order = seededOrder(n, 42);
        int nTrain = (int) (n * 0.7);
        int nTest = n - nTrain;
        float[] trainX = new float[nTrain * FEATURES];
        int[] trainY = new int[nTrain];
        float[] testX = new float[nTest * FEATURES];
        int[] testY = new int[nTest];
        for (int i = 0; i < n; i++) {
            if (i < nTrain) copyRow(x, y, order[i], trainX, trainY, i);
            else copyRow(x, y, order[i], testX, testY, i - nTrain);
        }
        System.out.printf("Split %d/%d - fraud in test set: %d%n",
                nTrain, nTest, countOf(testY, FRAUD));

        // ---------- 3. Standardize features with TRAIN statistics ---------
        float[] mean = columnMean(trainX, nTrain);
        float[] std = columnStd(trainX, nTrain, mean);
        standardize(trainX, mean, std);
        standardize(testX, mean, std);

        // ---------- 4. Build and train the MLP ----------------------------
        try (Sequential net = new Sequential(Optimizer.adam(0.005f),
                new Input(FEATURES),
                new Dense(32), new ReLU(),
                new BatchNorm1D(),               // keeps activations centered as classes are rare
                new Dense(16), new ReLU(),
                new Dense(2))) {

            net.fit(trainX, trainY, 120, 64, testX, testY, true);

            // ---------- 5. Evaluate at the default threshold -------------
            float[] probs = net.predict(testX);

            double loss = 0;
            int[][] conf = new int[2][2];
            for (int i = 0; i < nTest; i++) {
                int pred = Metrics.argmax(probs, i * 2, 2);   // implicit threshold P(fraud)>=0.5
                conf[testY[i]][pred]++;
                loss += -(testY[i] == 1
                        ? Math.log(Math.max(probs[i * 2 + 1], 1e-12))
                        : Math.log(Math.max(probs[i * 2], 1e-12)));
            }
            Metrics.printReport(loss / nTest, accuracy(conf), conf, NAMES);

            System.out.println("Majority baseline (always predict \"normal\"): "
                    + String.format("%.4f", (double) countOf(testY, 0) / nTest)
                    + " accuracy while finding ZERO frauds - why accuracy misleads here.");
            printFraudMetrics("argmax threshold (P(fraud)>=0.50)", conf);

            // ---------- 6. Threshold shift: catch more frauds -------------
            // Diagnostics: how far apart are the two populations in
            // probability space? If the lowest true-fraud score sits above
            // every true-normal score the classes separate cleanly.
            double minFraudP = 1.0, maxNormalP = 0.0;
            for (int i = 0; i < nTest; i++) {
                double pFraud = probs[i * 2 + 1];
                if (testY[i] == FRAUD) minFraudP = Math.min(minFraudP, pFraud);
                else maxNormalP = Math.max(maxNormalP, pFraud);
            }
            System.out.printf("Score separation: lowest P(fraud) among real frauds=%.4f,"
                    + " highest P(fraud) among normals=%.4f%n", minFraudP, maxNormalP);

            for (double t : new double[]{0.50, 0.35, 0.20}) {
                int[][] confShift = new int[2][2];
                for (int i = 0; i < nTest; i++) {
                    int pred = probs[i * 2 + 1] >= t ? 1 : 0;
                    confShift[testY[i]][pred]++;
                }
                printFraudMetrics(String.format(
                        java.util.Locale.ROOT, "threshold P(fraud)>=%.2f", t), confShift);
            }
            System.out.println("Lowering the threshold trades precision for recall -"
                    + " acceptable when a missed fraud costs far more than a review.");

            net.saveTo(Path.of("target", "fraud-model.bin"));
            System.out.println("Model saved to target/fraud-model.bin");
        }
    }

    /* ------------------------- data generation -------------------------- */

    /** Log-normal amount: median ~40 USD, scaled up for fraudulent rows. */
    static double lognormalAmount(Random r, double scale) {
        return Math.max(2.0, Math.exp(Math.log(40) + r.nextGaussian() * 0.75) * scale);
    }

    /** Daytime-weighted hour: |N(0,1)|*6 shifted by 8h wraps into night rarely. */
    static int daytimeHour(Random r) {
        return (int) (Math.abs(r.nextGaussian()) * 6 + 8) % 24;
    }

    /** Fraudsters work nights: hours drawn from 22..23 and 00..05 only. */
    static int oddHour(Random r) {
        return (r.nextInt(8) + 22) % 24;
    }

    static void writeRow(float[] x, int[] y, int row, double amount, int hour,
                         double dist, int online, float ratio, int label) {
        x[row * FEATURES] = (float) amount;
        x[row * FEATURES + 1] = hour;
        x[row * FEATURES + 2] = (float) dist;
        x[row * FEATURES + 3] = online;
        x[row * FEATURES + 4] = ratio;
        y[row] = label;
    }

    /* --------------------- metrics / split helpers ---------------------- */

    static void printFraudMetrics(String tag, int[][] conf) {
        double tp = conf[1][1], fn = conf[1][0], fp = conf[0][1];
        double recall = tp / Math.max(tp + fn, 1e-9);
        double precision = tp / Math.max(tp + fp, 1e-9);
        double f1 = 2 * precision * recall / Math.max(precision + recall, 1e-9);
        System.out.printf("%s -> fraud recall=%.3f precision=%.3f f1=%.3f (missed=%d, false alarms=%d)%n",
                tag, recall, precision, f1, (int) fn, (int) fp);
    }

    static double accuracy(int[][] conf) {
        int hit = 0, tot = 0;
        for (int[] row : conf) for (int c : row) tot += c;
        for (int k = 0; k < conf.length; k++) hit += conf[k][k];
        return (double) hit / tot;
    }

    static int countOf(int[] labels, int value) {
        int c = 0;
        for (int v : labels) if (v == value) c++;
        return c;
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

    static void copyRow(float[] src, int[] srcY, int row, float[] dst, int[] dstY, int dstRow) {
        System.arraycopy(src, row * FEATURES, dst, dstRow * FEATURES, FEATURES);
        dstY[dstRow] = srcY[row];
    }

    static float[] columnMean(float[] d, int rows) {
        float[] m = new float[FEATURES];
        for (int i = 0; i < rows; i++)
            for (int f = 0; f < FEATURES; f++) m[f] += d[i * FEATURES + f];
        for (int f = 0; f < FEATURES; f++) m[f] /= rows;
        return m;
    }

    static float[] columnStd(float[] d, int rows, float[] mean) {
        float[] s = new float[FEATURES];
        for (int i = 0; i < rows; i++)
            for (int f = 0; f < FEATURES; f++) {
                float diff = d[i * FEATURES + f] - mean[f];
                s[f] += diff * diff;
            }
        for (int f = 0; f < FEATURES; f++)
            s[f] = (float) Math.sqrt(s[f] / Math.max(rows - 1, 1));
        return s;
    }

    static void standardize(float[] d, float[] mean, float[] std) {
        for (int i = 0; i < d.length / FEATURES; i++)
            for (int f = 0; f < FEATURES; f++)
                d[i * FEATURES + f] = (d[i * FEATURES + f] - mean[f]) / (std[f] + 1e-8f);
    }

    private FraudDetection() {}
}
