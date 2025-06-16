package com.example.llmapi.service;

import com.example.llmapi.model.GeneratedRecipe;
import com.example.llmapi.util.FileUtil;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class RecipeService {

    public String getUserFridge(int userId) {
        String url = "http://localhost:8082/api/fridge/user/" + userId;
        RestTemplate restTemplate = new RestTemplate();
        return restTemplate.getForObject(url, String.class);
    }

    public void saveLlmRecipeLocally(int userId, GeneratedRecipe recipe) {
        FileUtil.saveGeneratedRecipeToJson(userId, recipe);
    }

    public void sendRecipeToMainApi(GeneratedRecipe recipe, int userId) {
        RestTemplate restTemplate = new RestTemplate();
        String url = "http://localhost:8080/api/recipe/receive/" + userId;

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<GeneratedRecipe> request = new HttpEntity<>(recipe, headers);

        try {
            restTemplate.postForObject(url, request, String.class);
        } catch (Exception e) {
            System.err.println("MainAPI'ye gönderim başarısız: " + e.getMessage());
        }
    }
}
