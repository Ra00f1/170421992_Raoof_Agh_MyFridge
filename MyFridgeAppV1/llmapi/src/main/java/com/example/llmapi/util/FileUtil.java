package com.example.llmapi.util;

import com.example.llmapi.model.GeneratedRecipe;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

public class FileUtil {

    public static void saveGeneratedRecipeToJson(int userId, GeneratedRecipe recipe) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String filename = "recipes/user_" + userId + "_recipe_" + timestamp + ".json";

            File file = new File(filename);
            file.getParentFile().mkdirs(); // klasörü oluştur
            mapper.writerWithDefaultPrettyPrinter().writeValue(file, recipe);

            System.out.println("Yemek tarifi kaydedildi: " + filename);
        } catch (Exception e) {
            System.err.println("Tarif kaydedilemedi: " + e.getMessage());
        }
    }
}
