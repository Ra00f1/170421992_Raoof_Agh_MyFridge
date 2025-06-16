package com.example.my_fridge_android.ui.home

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExitToApp
import androidx.compose.material.icons.filled.List
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.tooling.preview.PreviewParameter
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.my_fridge_android.ui.components.LoadingBar
import com.example.my_fridge_android.ui.home.HomeContract.UiAction
import com.example.my_fridge_android.ui.home.HomeContract.UiEffect
import com.example.my_fridge_android.ui.home.HomeContract.UiState
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emptyFlow

@Composable
fun HomeScreen(
    uiState: UiState,
    uiEffect: Flow<UiEffect>,
    onAction: (UiAction) -> Unit,
) {
    val snackbarHostState = remember { SnackbarHostState() }

    // Handle UI effects
    LaunchedEffect(uiEffect) {
        uiEffect.collect { effect ->
            when (effect) {
                is UiEffect.NavigateToReceiptList -> {
                    // Navigation handled by parent
                }

                is UiEffect.NavigateToRecipeChat -> {
                    // Navigation handled by parent
                }

                is UiEffect.NavigateToRecipeChatWithResponse -> {
                    // Navigation handled by parent
                }

                is UiEffect.NavigateToLogin -> {
                    // Navigation handled by parent
                }

                is UiEffect.ShowMessage -> {
                    snackbarHostState.showSnackbar(
                        message = effect.message,
                        duration = SnackbarDuration.Long
                    )
                }
            }
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        when {
            uiState.isLoading -> LoadingBar()
            else -> HomeContent(
                userName = uiState.userName,
                userId = uiState.userId,
                isSendingToRecipeAssistant = uiState.isSendingToRecipeAssistant,
                onNavigateToReceiptList = { onAction(UiAction.NavigateToReceiptList) },
                onSendItemsToRecipeAssistant = { onAction(UiAction.SendItemsToRecipeAssistant) },
                onNavigateToRecipeChat = { onAction(UiAction.NavigateToRecipeChat) },
                onLogout = { onAction(UiAction.Logout) }
            )
        }

        // Snackbar for messages
        SnackbarHost(
            hostState = snackbarHostState,
            modifier = Modifier.align(Alignment.BottomCenter)
        )
    }
}

@Composable
fun HomeContent(
    userName: String,
    userId: Int,
    isSendingToRecipeAssistant: Boolean,
    onNavigateToReceiptList: () -> Unit,
    onSendItemsToRecipeAssistant: () -> Unit,
    onNavigateToRecipeChat: () -> Unit,
    onLogout: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.2f),
                        MaterialTheme.colorScheme.surface
                    )
                )
            )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            // Header with logout button
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Welcome back!",
                        fontSize = 16.sp,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                    )
                    Text(
                        text = userName,
                        fontSize = 28.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    if (userId != -1) {
                        Text(
                            text = "User ID: $userId",
                            fontSize = 14.sp,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                        )
                    }
                }
                
                IconButton(
                    onClick = onLogout
                ) {
                    Icon(
                        imageVector = Icons.Default.ExitToApp,
                        contentDescription = "Logout",
                        tint = MaterialTheme.colorScheme.onSurface
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Main content
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "🍃 My Fridge",
                    fontSize = 48.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
                
                Text(
                    text = "Keep track of your groceries and get recipe suggestions",
                    fontSize = 16.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )

                Spacer(modifier = Modifier.height(32.dp))

                // Recipe Assistant Button
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(80.dp),
                    onClick = onSendItemsToRecipeAssistant,
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    ),
                    elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            text = "🤖",
                            fontSize = 32.sp
                        )

                        Column(
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(
                                text = "Recipe Assistant",
                                fontSize = 20.sp,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                            Text(
                                text = "Send your fridge items to Recipe Assistant",
                                fontSize = 14.sp,
                                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                            )
                        }

                        if (isSendingToRecipeAssistant) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(20.dp),
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                        } else {
                            Text(
                                text = "💬",
                                fontSize = 24.sp
                            )
                        }
                    }
                }

                // Recipe Chat Button  
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(80.dp),
                    onClick = onNavigateToRecipeChat,
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.tertiaryContainer
                    ),
                    elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            text = "💬",
                            fontSize = 32.sp
                        )

                        Column(
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(
                                text = "Recipe Chat",
                                fontSize = 20.sp,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onTertiaryContainer
                            )
                            Text(
                                text = "Chat with AI for recipe suggestions",
                                fontSize = 14.sp,
                                color = MaterialTheme.colorScheme.onTertiaryContainer.copy(alpha = 0.7f)
                            )
                        }

                        Text(
                            text = "🍳",
                            fontSize = 24.sp
                        )
                    }
                }

                // Receipt List Button
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(80.dp),
                    onClick = onNavigateToReceiptList,
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    ),
                    elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            text = "📋",
                            fontSize = 32.sp
                        )
                        
                        Column(
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(
                                text = "My Receipts",
                                fontSize = 20.sp,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                            Text(
                                text = "View and manage your receipts",
                                fontSize = 14.sp,
                                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                            )
                        }
                        
                        Icon(
                            imageVector = Icons.Default.List,
                            contentDescription = "Go to receipts",
                            tint = MaterialTheme.colorScheme.onPrimaryContainer,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                }

                Spacer(modifier = Modifier.weight(1f))

                // App info
                Text(
                    text = "Manage your fridge inventory with ease",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun HomeScreenPreview(
    @PreviewParameter(HomeScreenPreviewProvider::class) uiState: UiState,
) {
    MaterialTheme {
        HomeScreen(
            uiState = uiState,
            uiEffect = emptyFlow(),
            onAction = {},
        )
    }
}
