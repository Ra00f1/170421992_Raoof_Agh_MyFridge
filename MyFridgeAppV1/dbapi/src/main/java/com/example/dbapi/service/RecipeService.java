package com.example.dbapi.service;

import com.example.dbapi.model.Recipe;
import com.example.dbapi.model.UserRecipe;
import com.example.dbapi.repository.RecipeRepository;
import com.example.dbapi.repository.UserRecipeRepository;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Optional;

@Service
public class RecipeService {

    private final RecipeRepository recipeRepository;
    private final UserRecipeRepository userRecipeRepository;
    private final RestTemplate restTemplate;

    public RecipeService(RecipeRepository recipeRepository,
                         UserRecipeRepository userRecipeRepository,
                         RestTemplate restTemplate) {
        this.recipeRepository = recipeRepository;
        this.userRecipeRepository = userRecipeRepository;
        this.restTemplate = restTemplate;
    }

    public Recipe addRecipeForUser(Recipe incomingRecipe, Long userId) 
    {
        // Tarif sistemde var mı
        Optional<Recipe> existingRecipe = recipeRepository.findByYemekAdi(incomingRecipe.getYemekAdi());
        Recipe recipeToUse = existingRecipe.orElseGet(() -> recipeRepository.save(incomingRecipe));

        // Kullanıcıda bu tarif var mı
        Optional<UserRecipe> userRecipeOpt = userRecipeRepository.findByUserIdAndRecipe(userId, recipeToUse);

        if (userRecipeOpt.isPresent()) 
        {
            UserRecipe userRecipe = userRecipeOpt.get();
            userRecipe.setKullanimSayisi(userRecipe.getKullanimSayisi() + 1);
            userRecipeRepository.save(userRecipe);
        } 
        else 
        {
            UserRecipe newUserRecipe = new UserRecipe();
            newUserRecipe.setUserId(userId);
            newUserRecipe.setRecipe(recipeToUse);
            newUserRecipe.setKullanimSayisi(1);
            userRecipeRepository.save(newUserRecipe);
        }

        // Eksiltme işlemi
        try 
        {
            restTemplate.postForObject(
                "http://localhost:8080/api/recipe/deduct-ingredients/" + userId,
                incomingRecipe.getMalzemeler(), // List<String>
                String.class
            );
        } 
        catch (Exception e) 
        {
            System.err.println("Malzeme eksiltme isteği başarısız: " + e.getMessage());
        }

        return recipeToUse;
    }

    public void sendRecipeToMainApi(Recipe recipe, int userId) 
    {
        String url = "http://localhost:8080/api/recipe/receive/" + userId;
        restTemplate.postForObject(url, recipe, String.class);
    }
}
