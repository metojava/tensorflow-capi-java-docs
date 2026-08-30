package com.example.house;

import com.example.tfcapi.*;

import java.util.Arrays;
import java.util.Random;

/**
 * Project 3 - House price band classification (regression turned into
 * classification).
 *
 * Instead of predicting the exact price (regression), we bucket each house
 * into one of three price bands - cheap, medium, expensive - using tertiles
 * computed on TRAINING prices only. This demonstrates:
 *
 * 1. generating a synthetic tabular dataset with a fixed random seed,
 * 2. a seeded shuffle + 70/30 train/test split,
 * 3. standardizing features with training-set statistics,
 * 4. converting a numeric target into class labels via tertile thresholds,
 * 5. training an MLP and reading a per-class report (precision/recall/F1).
 */
public final class HousePriceClassifier {

    private static final int N = 1200;                 // synthetic houses
    private static final int FEATURES = 4;
    private static final int CLASSES = 3;
    private static final long SEED = 42L;

    private static final String[] BAND_NAMES = {"cheap", "medium", "expensive"};

    public static void main(String[] args) throws Exception {
        // ---------- 1. Generate the synthetic dataset --------------------
        Random r = new Random(SEED);
        float[] sqft = new float[N];
        int[] bedrooms = new int[N];
        int[] age = new int[N];
        int[] location = new int[N];
        double[] price = new double[N];
        for (int i = 0; i < N; i++) {
            sqft[i] = 500 + r.nextFloat() * 3000;              // 500 .. 3500
            bedrooms[i] = 1 + r.nextInt(6);                    // 1 .. 6
            age[i] = r.nextInt(81);                            // 0 .. 80
            location[i] = 1 + r.nextInt(10);                   // 1 .. 10
            // Ground-truth pricing formula plus Gaussian noise.
            double p = 120 * sqft[i] + 8000 * bedrooms[i] - 900 * age[i]
                     + 15000 * location[i]
                     + r.nextGaussian() * 25000;
            price[i] = Math.max(p, 50000);                     // clip below 50k
        }
        System.out.println("Generated " + N + " synthetic house records (seed "
                + SEED + ")");

        // ---------- 2. Seeded shuffle, then a 70/30 split ---------------
        int[] order = seededOrder(N, SEED);
        int trainN = (int) Math.round(N * 0.7);                // 840
        int testN = N - trainN;                                // 360

        float[][] splitX = {new float[trainN * FEATURES], new float[testN * FEATURES]};
        double[][] splitPrice = {new double[trainN], new double[testN]};
        for (int i = 0; i < N; i++) {
            int src = order[i];
            boolean isTrain = i < trainN;
            float[] dx = isTrain ? splitX[0] : splitX[1];
            double[] dp = isTrain ? splitPrice[0] : splitPrice[1];
            int di = isTrain ? i : i - trainN;
            dx[di * FEATURES]     = sqft[src];
            dx[di * FEATURES + 1] = bedrooms[src];
            dx[di * FEATURES + 2] = age[src];
            dx[di * FEATURES + 3] = location[src];
            dp[di] = price[src];
        }

        // ---------- 3. Price bands via tertiles of TRAIN prices ----------
        // Sorting only the training prices keeps test information out of the
        // labeling rule - exactly what you must do to avoid target leakage.
        double[] sortedTrainPrices = splitPrice[0].clone();
        Arrays.sort(sortedTrainPrices);
        double t1 = sortedTrainPrices[trainN / 3];             // ~33rd percentile
        double t2 = sortedTrainPrices[2 * trainN / 3];         // ~67th percentile
        System.out.printf("Tertile thresholds from train prices: %.0f / %.0f%n", t1, t2);
        int[] trainY = bandLabels(splitPrice[0], t1, t2);
        int[] testY = bandLabels(splitPrice[1], t1, t2);

        // ---------- 4. Standardize inputs with TRAIN statistics ----------
        float[] mean = columnMean(splitX[0], trainN);
        float[] std = columnStd(splitX[0], trainN, mean);
        standardize(splitX[0], mean, std);
        standardize(splitX[1], mean, std);

        // ---------- 5. Build and train the MLP ---------------------------
        try (Sequential net = new Sequential(Optimizer.adam(0.01f),
                new Input(FEATURES),
                new Dense(24), new ReLU(),
                new Dense(12), new ReLU(),
                new Dense(CLASSES))) {

            net.fit(splitX[0], trainY, 200, 32, splitX[1], testY, true);

            // ---------- 6. Evaluate: accuracy + full confusion report ----
            float[] probs = net.predict(splitX[1]);
            double loss = 0;
            int correct = 0;
            int[][] conf = new int[CLASSES][CLASSES];
            for (int i = 0; i < testN; i++) {
                int pred = Metrics.argmax(probs, i * CLASSES, CLASSES);
                conf[testY[i]][pred]++;
                if (pred == testY[i]) correct++;
                loss += -Math.log(Math.max(probs[i * CLASSES + testY[i]], 1e-12));
            }
            Metrics.printReport(loss / testN, (double) correct / testN, conf, BAND_NAMES);
        }
    }

    /* ------------------------- helper methods --------------------------- */

    /** Fisher-Yates shuffle of indices 0..n-1 with a fixed seed (reproducible). */
    static int[] seededOrder(int n, long seed) {
        int[] o = new int[n];
        for (int i = 0; i < n; i++) o[i] = i;
        Random rr = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = rr.nextInt(i + 1), t = o[i]; o[i] = o[j]; o[j] = t;
        }
        return o;
    }

    /**
     * Maps continuous prices to band labels using thresholds learned on the
     * training set: price &le; t1 -> cheap(0), &le; t2 -> medium(1),
     * else expensive(2).
     */
    static int[] bandLabels(double[] prices, double t1, double t2) {
        int[] y = new int[prices.length];
        for (int i = 0; i < prices.length; i++) {
            y[i] = prices[i] <= t1 ? 0 : (prices[i] <= t2 ? 1 : 2);
        }
        return y;
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

    private HousePriceClassifier() {}
}
