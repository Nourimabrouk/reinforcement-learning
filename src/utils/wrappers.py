def train_agent(agent, env, num_episodes, max_steps_per_episode, comet_experiment=None):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            agent.learn((state, action, reward, next_state, done))
            
            if done:
                break
                
            state = next_state
            
        if comet_experiment is not None:
            comet_experiment.log_metric("Episode Reward", episode_reward, step=episode)
        
        print(f"Episode {episode}: Reward = {episode_reward}")
        
    agent.save('agent.pkl')
