package com.example.my_fridge_android.di

import com.example.my_fridge_android.data.config.ServerConfig
import com.example.my_fridge_android.data.source.remote.AuthService
import com.example.my_fridge_android.data.source.remote.DeleteApiService
import com.example.my_fridge_android.data.source.remote.FridgeService
import com.example.my_fridge_android.data.source.remote.MainService
import com.example.my_fridge_android.data.source.remote.ManualIngredientService
import com.example.my_fridge_android.data.source.remote.RecipeAssistantService
import com.example.my_fridge_android.data.source.remote.ReduceAmountApiService

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit
import javax.inject.Qualifier
import javax.inject.Singleton
import com.google.gson.Gson
import com.google.gson.GsonBuilder

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class OcrRetrofit

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class AuthRetrofit

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class RecipeRetrofit

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class DeleteRetrofit

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class ReduceAmountRetrofit

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class ManualIngredientRetrofit

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    fun provideOkHttpClient(): OkHttpClient {
        return OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .addInterceptor { chain ->
                val request = chain.request().newBuilder()
                    .addHeader("Accept-Charset", "UTF-8")
                    .build()
                chain.proceed(request)
            }
            .build()
    }

    @Provides
    @Singleton
    fun provideGson(): Gson {
        return GsonBuilder()
            .setLenient()
            .serializeNulls()
            .create()
    }

    @Provides
    @Singleton
    @OcrRetrofit
    fun provideOcrRetrofit(
        okHttpClient: OkHttpClient,
        serverConfig: ServerConfig,
        gson: Gson
    ): Retrofit {
        return Retrofit.Builder()
            .baseUrl(serverConfig.baseUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
    }

    @Provides
    @Singleton
    @AuthRetrofit
    fun provideAuthRetrofit(
        okHttpClient: OkHttpClient,
        serverConfig: ServerConfig,
        gson: Gson
    ): Retrofit {
        return Retrofit.Builder()
            .baseUrl(serverConfig.authServerUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
    }

    @Provides
    @Singleton
    fun provideFridgeRetrofit(
        okHttpClient: OkHttpClient,
        serverConfig: ServerConfig,
        gson: Gson
    ): Retrofit {
        return Retrofit.Builder()
            .baseUrl(serverConfig.fridgeServerUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
    }

    @Provides
    @Singleton
    @RecipeRetrofit
    fun provideRecipeRetrofit(
        okHttpClient: OkHttpClient,
        serverConfig: ServerConfig,
        gson: Gson
    ): Retrofit {
        return Retrofit.Builder()
            .baseUrl(serverConfig.recipeAssistantServerUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
    }

    @Provides
    @Singleton
    @DeleteRetrofit
    fun provideDeleteRetrofit(
        okHttpClient: OkHttpClient,
        serverConfig: ServerConfig,
        gson: Gson
    ): Retrofit {
        return Retrofit.Builder()
            .baseUrl(serverConfig.deleteApiServerUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
    }

    @Provides
    @Singleton
    @ReduceAmountRetrofit
    fun provideReduceAmountRetrofit(
        okHttpClient: OkHttpClient,
        serverConfig: ServerConfig,
        gson: Gson
    ): Retrofit {
        return Retrofit.Builder()
            .baseUrl(serverConfig.reduceAmountApiServerUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
    }

    @Provides
    @Singleton
    @ManualIngredientRetrofit
    fun provideManualIngredientRetrofit(
        okHttpClient: OkHttpClient,
        serverConfig: ServerConfig,
        gson: Gson
    ): Retrofit {
        return Retrofit.Builder()
            .baseUrl(serverConfig.manualIngredientServerUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
    }

    @Provides
    fun provideMainService(@OcrRetrofit retrofit: Retrofit): MainService {
        return retrofit.create(MainService::class.java)
    }

    @Provides
    fun provideAuthService(@AuthRetrofit retrofit: Retrofit): AuthService {
        return retrofit.create(AuthService::class.java)
    }

    @Provides
    fun provideFridgeService(retrofit: Retrofit): FridgeService {
        return retrofit.create(FridgeService::class.java)
    }

    @Provides
    fun provideRecipeAssistantService(@RecipeRetrofit retrofit: Retrofit): RecipeAssistantService {
        return retrofit.create(RecipeAssistantService::class.java)
    }

    @Provides
    fun provideDeleteApiService(@DeleteRetrofit retrofit: Retrofit): DeleteApiService {
        return retrofit.create(DeleteApiService::class.java)
    }

    @Provides
    fun provideReduceAmountApiService(@ReduceAmountRetrofit retrofit: Retrofit): ReduceAmountApiService {
        return retrofit.create(ReduceAmountApiService::class.java)
    }

    @Provides
    fun provideManualIngredientService(@ManualIngredientRetrofit retrofit: Retrofit): ManualIngredientService {
        return retrofit.create(ManualIngredientService::class.java)
    }
}
