package com.example.housereg;

import com.example.tfcapi.*;

import java.nio.file.Path;
import java.util.Random;

/**
 * Project 11 - House price REGRESSION (predicting the exact sale price).
 *
 * Companion to house-price-classifier (project 3): same synthetic housing
 * market idea, but the target stays continuous and is learned with the
 * Regressor model (MSE loss, linear output). Demonstrates:
 *
 * 1. generating a seeded synthetic dataset with six features,
 * 2. Datasets.trainTestSplit + gatherRows/gatherValues for reproducible splits,
 * 3. standardizing features AND the target with training-split statistics,
 * 4. Regressor with seeded init/shuffle, cosine LR schedule,
 *    decoupled weight decay and global-norm gradient clipping,
 * 5. EarlyStopping.valLoss(25, true) deciding the epoch count,
 * 6. reporting RMSE / MAE / R-squared plus sample predictions in dollars.
 */
public final class HousePriceRegressor {

    private static final int N = 1600;                 // synthetic houses
                                                       // (70% train split =
                                                       // 1120 = 35 x batch 32)
    private static final int F = 6;                    // feature count
    private static final long SEED = 42L;

    public static void main(String[] args) throws Exception {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 300;

        // ---------- 1. Generate the synthetic market ----------------------
        Random r = new Random(SEED);
        float[] X = new float[N * F];
        float[] price = new float[N];
        for (int i = 0; i < N; i++) {
            float sqft = 500 + r.nextFloat() * 3000;          // 500 .. 3500
            int bedrooms = 1 + r.nextInt(6);                  // 1 .. 6
            int age = r.nextInt(81);                          // 0 .. 80
            int location = 1 + r.nextInt(10);                 // 1 .. 10
            int garage = r.nextInt(4);                        // 0 .. 3
            float distKm = 0.5f + r.nextFloat() * 29.5f;      // 0.5 .. 30

            double p = 120 * sqft
                     + 8000 * bedrooms
                     - 900 * age
                     + 15000 * location
                     + 12000 * garage
                     - 2500 * distKm
                     + 8 * sqft * location                    // interaction:
                     + r.nextGaussian() * 25000;              // premium districts
            X[i * F]     = sqft;
            X[i * F + 1] = bedrooms;
            X[i * F + 2] = age;
            X[i * F + 3] = location;
            X[i * F + 4] = garage;
            X[i * F + 5] = distKm;
            price[i] = (float) Math.max(p, 50000);
        }
        System.out.println("Generated " + N + " synthetic house records (seed "
                + SEED + ")");

        // ---------- 2. Seeded split via Datasets --------------------------
        int[][] parts = Datasets.trainTestSplit(N, 0.3, SEED);
        int[] trIdx = parts[0], teIdx = parts[1];
        float[] trainX = Datasets.gatherRows(X, trIdx, F);
        float[] testX = Datasets.gatherRows(X, teIdx, F);
        float[] trainP = gatherFloats(price, trIdx);
        float[] testP = gatherFloats(price, teIdx);
        System.out.printf("Split: %d train / %d test%n",
                trIdx.length, teIdx.length);

        // ---------- 3. Standardize features and target (train stats!) -----
        float[][] norm = Datasets.fitStandardizer(trainX, F);
        Datasets.applyStandardizer(trainX, norm[0], norm[1]);
        Datasets.applyStandardizer(testX, norm[0], norm[1]);

        float yMean = mean(trainP);
        float yStd = std(trainP, yMean);
        float[] trainY = zscore(trainP, yMean, yStd);
        float[] testY = zscore(testP, yMean, yStd);

        // ---------- 4. Regularized optimizer ------------------------------
        Optimizer opt = Optimizer.adam(0.01f);
        opt.setWeightDecay(1e-4f);     // decoupled (AdamW-style) decay
        opt.setGradClipNorm(5f);       // safety net against gradient spikes

        // ---------- 5. Build and train the Regressor -----------------------
        try (Regressor net = new Regressor(opt, SEED,
                new Input(F),
                new Dense(64), new ReLU(),
                new Dense(32), new ReLU(),
                new Dense(1))) {

            net.setShuffleSeed(SEED);
            net.setLrSchedule(LrSchedule.cosine(epochs, 0.01f));

            EarlyStopping es = EarlyStopping.valLoss(25, true);
            Regressor.RegressionHistory h =
                    net.fit(trainX, trainY, epochs, 32, testX, testY, true, es);

            System.out.printf("Ran %d/%d epochs - best val MSE %.5f at epoch %d%n",
                    h.epochsRun(), epochs, es.bestScore, es.bestEpoch + 1);

            // ---------- 6. Evaluate in DOLLARS ----------------------------
            float[] predZ = net.predict(testX);
            double ssRes = 0, ssTot = 0, absSum = 0;
            double pBar = mean(testP);
            for (int i = 0; i < testP.length; i++) {
                double pred = predZ[i] * yStd + yMean;   // back to dollars
                double e = pred - testP[i];
                absSum += Math.abs(e);
                ssRes += e * e;
                ssTot += (testP[i] - pBar) * (testP[i] - pBar);
            }
            int n = testP.length;
            double rmse = Math.sqrt(ssRes / n);
            double mae = absSum / n;
            double r2 = 1 - ssRes / ssTot;
            System.out.printf("RMSE $%,.0f | MAE $%,.0f | R^2 %.3f%n%n",
                    rmse, mae, r2);

            System.out.println("Sample predictions:");
            System.out.println("     actual      predicted");
            for (int i = 0; i < 8 && i < n; i++) {
                double pred = predZ[i] * yStd + yMean;
                System.out.printf("%2d) $%,9.0f  $%,9.0f%n",
                        i + 1, testP[i], pred);
            }

            net.saveTo(Path.of("target", "house-regressor.bin"));
            System.out.println("Saved checkpoint target/house-regressor.bin");
        }
    }

    /* ------------------------- helper methods --------------------------- */

    /** Copies values at the given indices (float twin of Datasets.gatherValues). */
    static float[] gatherFloats(float[] values, int[] indices) {
        float[] out = new float[indices.length];
        for (int i = 0; i < indices.length; i++) out[i] = values[indices[i]];
        return out;
    }

    static float mean(float[] v) {
        double s = 0;
        for (double x : v) s += x;
        return (float) (s / v.length);
    }

    static float std(float[] v, float m) {
        double s = 0;
        for (double x : v) s += (x - m) * (x - m);
        return (float) Math.sqrt(s / Math.max(v.length - 1, 1));
    }

    static float[] zscore(float[] v, float m, float s) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = (v[i] - m) / (s + 1e-8f);
        return out;
    }

    private HousePriceRegressor() {}
}
