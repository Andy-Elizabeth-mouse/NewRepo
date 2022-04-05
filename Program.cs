using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;



#pragma warning disable CS8321 //声明了本地函数，但从未使用过

public static class AgentModels
{
    static readonly Tensorflow.Keras.Layers.LayersApi layers = keras.layers;
    public static Model Policy_Model(int state_dim, int action_dim, int hidden_dim = 256)
    {
        var inp = layers.Input(state_dim);
        var d1 = layers.Dense(hidden_dim).Apply(inp);
        var d2 = layers.Dense(hidden_dim).Apply(d1);
        var act = layers.Dense(action_dim).Apply(d2);
        var v = layers.Dense(action_dim).Apply(d2);
        return keras.Model(inp, new Tensors(act, v));
    }

    public static Model Value_Model(int state_dim, int action_dim, int hidden_dim = 256)
    {
        var inp = layers.Input(state_dim);
        var d1 = layers.Dense(hidden_dim).Apply(inp);
        var d2 = layers.Dense(hidden_dim).Apply(d1);
        return keras.Model(inp, layers.Dense(action_dim).Apply(d2));
    }
}

public class Discrete
{
    public Tensor sample(Tensor datas)
    {
        //distribution = tfp.distributions.Categorical(probs = datas)
        //return distribution.sample()
        return tf.random.categorical(tf.math.log(datas), 1);
    }

    public Tensor entropy(Tensor datas)
    {
        //distribution = tfp.distributions.Categorical(probs = datas)            
        //return distribution.entropy()
        return -tf.reduce_sum(tf.multiply(tf.math.log(datas), datas), axis: -1);
    }

    public Tensor logprob(Tensor datas, Tensor value_data)
    {
        //distribution = tfp.distributions.Categorical(probs = datas)
        //return tf.expand_dims(distribution.log_prob(value_data), 1)
        value_data = tf.cast(value_data, tf.int32);
        var result = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels: value_data, logits: tf.math.log(datas));
        return tf.expand_dims(result, 1);
    }

    public Tensor kl_divergence(Tensor datas1, Tensor datas2)
    {
        //distribution1 = tfp.distributions.Categorical(probs = datas1)
        //distribution2 = tfp.distributions.Categorical(probs = datas2) 
        //return tf.expand_dims(tfp.distributions.kl_divergence(distribution1, distribution2), 1)
        var a_logits = tf.math.log(datas1);
        var b_logits = tf.math.log(datas2);
        var result = tf.reduce_sum(tf.multiply(
                                    tf.math.log(tf.nn.softmax(a_logits)) -
                                    tf.math.log(tf.nn.softmax(b_logits)), tf.nn.softmax(a_logits)), axis: -1);
        return tf.expand_dims(result, 1);
    }
}

public class Memory
{
    public class TensorSliceDataset
    {
        Tensors _tensors;
        public int Count;
        public TensorSliceDataset(params Tensor[] datas)
        {
            Count = (int)datas[0].shape[0];
            _tensors = datas;
        }
        public IEnumerable<Tensors> batch(int batch_size)
        {
            for(int i = 0; i < Count / batch_size; i++)
            {
                var index = i * batch_size;
                var begin = new[] { index, 0 };
                var size = new[] { batch_size, -1 };
                Tensors outp = new();
                foreach (var tensor in _tensors)
                    outp.Add(tf.expand_dims(tf.slice(outp, begin, size)));
                yield return outp;
            }
        }
    }
    public List<Tensor> actions;
    public List<Tensor> dones;
    public List<Tensor> next_states;
    public List<Tensor> rewards;
    public List<Tensor> states;
    public Memory()
    {
        this.actions = new List<Tensor>();
        this.states = new List<Tensor>();
        this.rewards = new List<Tensor>();
        this.dones = new List<Tensor>();
        this.next_states = new List<Tensor>();
    }
    public int length { get { return actions.Count; } }
    public TensorSliceDataset get_all_tensor()
    {
        var states = tf.constant(this.states, dtype: tf.float32);
        var actions = tf.constant(this.actions, dtype: tf.float32);
        var rewards = tf.expand_dims(tf.constant(this.rewards, dtype: tf.float32), 1);
        var dones = tf.expand_dims(tf.constant(this.dones, dtype: tf.float32), 1);
        var next_states = tf.constant(this.next_states, dtype: tf.float32);
        return new TensorSliceDataset(states, actions, rewards, dones, next_states);
    }
    public (List<Tensor>, List<Tensor>, List<Tensor>, List<Tensor>, List<Tensor>) get_all()
    {
        return (this.states, this.actions, this.rewards, this.dones, this.next_states);
    }
    public void save_eps(
        Tensor state,
        Tensor action,
        Tensor reward,
        Tensor done,
        Tensor next_state)
    {
        this.rewards.append(reward);
        this.states.append(state);
        this.actions.append(action);
        this.dones.append(done);
        this.next_states.append(next_state);
    }
    public virtual void clear_memory()
    {
        this.actions.Clear();
        this.states.Clear();
        this.rewards.Clear();
        this.dones.Clear();
        this.next_states.Clear();
    }
}

public class PPOLoss
{

    public Discrete distributions;

    public float entropy_coef;

    public float policy_kl_range;

    public float policy_params;

    public float value_clip;

    public float vf_loss_coef;

    public float gamma;

    public float lam;
    public PPOLoss(
        float policy_kl_range,
        float policy_params,
        float value_clip,
        float vf_loss_coef,
        float entropy_coef,
        float gamma,
        float lam)
    {
        this.policy_kl_range = policy_kl_range;
        this.policy_params = policy_params;
        this.value_clip = value_clip;
        this.vf_loss_coef = vf_loss_coef;
        this.entropy_coef = entropy_coef;
        this.distributions = new Discrete();
        this.gamma = gamma;
        this.lam = lam;
        //this.policy_function = new PolicyFunction(gamma, lam);
    }

    public Tensor compute_gae(Tensor values, Tensor rewards, Tensor next_values, Tensor dones)
    {
        Tensor gae = new Tensor(0);
        var adv = new Tensor[rewards.shape[^1]];
        var delta = rewards + (1.0 - dones) * this.gamma * next_values - values;
        for (var step = (int)rewards.shape[^1]; step >= 0; step--)
        {
            gae = delta[step] + (1f - dones[step]) * this.gamma * this.lam * gae;
            adv[step] = gae;
        }
        return tf.stack(adv);
    }

    // Loss for PPO  
    public Tensor compute_loss(
        Tensor action_probs,
        Tensor old_action_probs,
        Tensor values,
        Tensor old_values,
        Tensor next_values,
        Tensor actions,
        Tensor rewards,
        Tensor dones)
    {
        // Don't use old value in backpropagation
        var Old_values = tf.stop_gradient(old_values);
        var Old_action_probs = tf.stop_gradient(old_action_probs);

        // Getting general advantages estimator
        var Advantages = this.compute_gae(values, rewards, next_values, dones);
        var Returns = tf.stop_gradient(Advantages + values);
        Advantages = tf.stop_gradient((Advantages - tf.reduce_mean(Advantages)) / (tf.reduce_std(Advantages) + 1E-06f));

        // Finding the ratio (pi_theta / pi_theta__old):        
        var logprobs = this.distributions.logprob(action_probs, actions);
        var Old_logprobs = tf.stop_gradient(this.distributions.logprob(Old_action_probs, actions));
        var ratios = tf.exp(logprobs - Old_logprobs);

        // Finding KL Divergence                
        var Kl = this.distributions.kl_divergence(Old_action_probs, action_probs);

        // Combining TR-PPO with Rollback (Truly PPO)
        var pg_loss = tf.where(tf.logical_and(Kl >= this.policy_kl_range, ratios > 1), ratios * Advantages - this.policy_params * Kl, ratios * Advantages);
        pg_loss = tf.reduce_mean(pg_loss);

        // Getting entropy from the action probability
        var dist_entropy = tf.reduce_mean(this.distributions.entropy(action_probs));

        // Getting critic loss by using Clipped critic value
        var vpredclipped = old_values + tf.clip_by_value(values - Old_values, -this.value_clip, this.value_clip);
        var vf_losses1 = tf.square(Returns - values) * 0.5;
        var vf_losses2 = tf.square(Returns - vpredclipped) * 0.5;
        var critic_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2));

        // We need to maximaze Policy Loss to make agent always find Better Rewards
        // and minimize Critic Loss 
        var loss = critic_loss * this.vf_loss_coef - dist_entropy * this.entropy_coef - pg_loss;
        return loss;
    }
}

public class AuxLoss
{

    public Discrete distributions;

    public AuxLoss()
    {
        this.distributions = new Discrete();
    }

    public Tensor compute_loss(Tensor action_probs, Tensor old_action_probs, Tensor values, Tensor Returns)
    {
        // Don't use old value in backpropagation
        var Old_action_probs = tf.stop_gradient(old_action_probs);
        // Finding KL Divergence                
        var Kl = tf.reduce_mean(this.distributions.kl_divergence(Old_action_probs, action_probs));
        var aux_loss = tf.reduce_mean(tf.square(Returns - values) * 0.5);
        return aux_loss + Kl;
    }
}

public class Agent
{

    public int action_dim;

    public AuxLoss aux_loss;

    public int batchsize;

    public Discrete distributions;

    public float entropy_coef;

    public bool is_training_mode;

    public Tensorflow.Keras.Optimizers.OptimizerV2 optimizer;

    public float policy_kl_range;

    public PPOLoss policy_loss;

    public Memory policy_memory;

    public object policy_params;

    public int PPO_epochs;

    public float value_clip;
    
    public float vf_loss_coef;

    public Model policy;

    public Model policy_old;

    public Model value;

    public Model value_old;

    public Agent(
        int state_dim,
        int action_dim,
        bool is_training_mode,
        float policy_kl_range,
        int policy_params,
        float value_clip,
        float entropy_coef,
        float vf_loss_coef,
        int batchsize,
        int PPO_epochs,
        float gamma,
        float lam,
        float learning_rate)
    {
        this.policy_kl_range = policy_kl_range;
        this.policy_params = policy_params;
        this.value_clip = value_clip;
        this.entropy_coef = entropy_coef;
        this.vf_loss_coef = vf_loss_coef;
        this.batchsize = batchsize;
        this.PPO_epochs = PPO_epochs;
        this.is_training_mode = is_training_mode;
        this.action_dim = action_dim;
        this.policy = AgentModels.Policy_Model(state_dim, action_dim);
        this.policy_old = AgentModels.Policy_Model(state_dim, action_dim);
        this.value = AgentModels.Value_Model(state_dim, action_dim);
        this.value_old = AgentModels.Value_Model(state_dim, action_dim);
        this.policy_memory = new Memory();
        this.policy_loss = new PPOLoss(policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma, lam);
        this.aux_loss = new AuxLoss();
        this.optimizer = keras.optimizers.Adam(learning_rate: learning_rate);
        this.distributions = new Discrete();
    }

    public void save_eps(
        Tensor state,
        Tensor action,
        Tensor reward,
        Tensor done,
        Tensor next_state) =>
            this.policy_memory.save_eps(state, action, reward, done, next_state);

    public Tensor act(Tensor state)
    {
        state = tf.expand_dims(tf.cast(state, dtype: tf.float32), 0);
        var action_probs = this.policy.Apply(state);
        action_probs = action_probs[0];
        // We don't need sample the action in Test Mode
        // only sampling the action in Training Mode in order to exploring the actions
        return this.is_training_mode ? this.distributions.sample(action_probs)
                                     : tf.math.argmax(action_probs, 1);
    }

    // Get loss and Do backpropagation
    [Tensorflow.Graphs.AutoGraph]
    public void training_ppo(
        Tensor states,
        Tensor actions,
        Tensor rewards,
        Tensor dones,
        Tensor next_states)
    {
        using var tape = tf.GradientTape() ;
            var action_probs = this.policy.Apply(states)[0];
            var values = this.value.Apply(states);
            var old_action_probs = this.policy_old.Apply(states)[0];
            var old_values = this.value_old.Apply(states);
            var next_values = this.value.Apply(next_states);
            var loss = this.policy_loss.compute_loss(action_probs, old_action_probs, values, old_values, next_values, actions, rewards, dones);

        var variables = new List<IVariableV1>();
        variables.AddRange(this.policy.trainable_variables);
        variables.AddRange(this.value.trainable_variables);
        var gradients = tape.gradient(loss, variables);
        this.optimizer.apply_gradients((IEnumerable<(Tensor, ResourceVariable)>)zip(gradients, variables));
    }

    [Tensorflow.Graphs.AutoGraph]
    public void training_aux(Tensor states)
    {
        var Returns = tf.stop_gradient(this.value.Apply(states));
        using var tape = tf.GradientTape();
            var (action_probs, values) = this.policy.Apply(states);
            var old_action_probs = this.policy_old.Apply(states)[0];
            var joint_loss = this.aux_loss.compute_loss(action_probs, old_action_probs, values, Returns);
        
        var gradients = tape.gradient(joint_loss, this.policy.trainable_variables);
        this.optimizer.apply_gradients((IEnumerable<(Tensor, ResourceVariable)>)zip(gradients, this.policy.trainable_variables));
    }

    // Update the model
    public void update_ppo()
    {
        Tensor states;
        foreach (var _ in Enumerable.Range(0, this.PPO_epochs))
        {
            foreach (var datas in this.policy_memory.get_all_tensor().batch(this.batchsize))
            {
                states = datas[0];
                var actions = datas[1];
                var rewards = datas[2];
                var dones = datas[3];
                var next_states = datas[4];
                this.training_ppo(states, actions, rewards, dones, next_states);
            }
        }
        // Clear the memory
        //var _tup_2 = this.policy_memory.get_all();
        //this.policy_memory.clear_memory();
        // Copy new weights into old policy:
        //self.policy_old.set_weights(self.policy.get_weights())
        //self.value_old.set_weights(self.value.get_weights())
        
        //soft update
        //var w_new = this.policy.weights;
        //var w_old = this.policy_old.weights;
        //Console.WriteLine($"{len(w_new)},{len(w_old)}");
        //foreach (var _tup_3 in zip(w_new, w_old))
        //{
        //    var w_new_ = _tup_3.Item1;
        //    var w_old_ = _tup_3.Item2;
        //    w_new_ = 0.7f * w_new_ + 0.3f * w_old_;
        //}
        //Console.WriteLine($"{len(w_new)},{len(w_old)}");
        //this.policy_old.set_weights(w_new);
        //w_new = this.value.get_weights();
        //w_old = this.value_old.get_weights();
        //foreach (var _tup_4 in zip(w_new, w_old))
        //{
        //    w_new_ = _tup_4.Item1;
        //    w_old_ = _tup_4.Item2;
        //    w_new_ = 0.7 * w_new_ + 0.3 * w_old_;
        //}
        //this.policy_old.set_weights(w_old);
    }

    public void update_aux()
    {
        // Optimize policy for K epochs:
        foreach (var _ in Enumerable.Range(0, this.PPO_epochs))
        {
            foreach (var states in this.aux_memory.get_all_tensor().batch(this.batchsize))
            {
                this.training_aux(states);
            }
        }
        // Clear the memory
        this.aux_memory.clear_memory();
        // Copy new weights into old policy:
        //self.policy_old.set_weights(self.policy.get_weights()) 
        var w_new = this.policy.get_weights();
        var w_old = this.policy_old.get_weights();
        foreach (var _tup_1 in zip(w_new, w_old))
        {
            var w_new_ = _tup_1.Item1;
            var w_old_ = _tup_1.Item2;
            w_new_ = 0.7 * w_new_ + 0.3 * w_old_;
        }
        this.policy_old.set_weights(w_new);
    }

    public virtual void save_weights()
    {
        this.policy.save_weights("bipedalwalker_w/policy_ppo", save_format: "tf");
        this.value.save_weights("bipedalwalker_w/critic_ppo", save_format: "tf");
    }

    public virtual void load_weights()
    {
        this.policy.load_weights("bipedalwalker_w/policy_ppo");
        this.value.load_weights("bipedalwalker_w/value_ppo");
    }
}

